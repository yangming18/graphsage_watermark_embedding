import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import random
from torch.optim.lr_scheduler import LinearLR


# MeanAggregator class with neighbor sampling
class MeanAggregator(nn.Module):
    def __init__(self, num_sample=10):
        super(MeanAggregator, self).__init__()
        self.num_sample = num_sample

    def forward(self, nodes, adj_lists, features):
        agg_feats = []
        for node in nodes:
            neighbor_ids = adj_lists[node.item()]
            if len(neighbor_ids) == 0:
                neighbor_feats = features[node].unsqueeze(0)
            else:
                if len(neighbor_ids) > self.num_sample:
                    sampled_neighbors = random.sample(neighbor_ids, self.num_sample)
                else:
                    sampled_neighbors = neighbor_ids
                neighbor_feats = features[sampled_neighbors]
            agg_feats.append(neighbor_feats.mean(dim=0))
        return torch.stack(agg_feats)

# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, aggregator, num_sample=10, gcn=False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.gcn = gcn

        if self.gcn:
            self.linear = nn.Linear(input_dim, embed_dim)
        else:
            self.linear = nn.Linear(2 * input_dim, embed_dim)

    def forward(self, nodes, features, edge_index):
        node_adj_lists = self.get_neighbors(nodes, edge_index)
        neigh_feats = self.aggregator(nodes, node_adj_lists, features)

        if self.gcn:
            combined = neigh_feats
        else:
            self_feats = features[nodes]
            combined = torch.cat([self_feats, neigh_feats], dim=1)

        combined = self.linear(combined)
        combined = F.relu(combined)
        return combined

    def get_neighbors(self, nodes, edge_index):
        neighbors = {node.item(): [] for node in nodes}
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        for src, dst in zip(src_nodes, dst_nodes):
            src = src.item()
            dst = dst.item()
            if src in neighbors:
                neighbors[src].append(dst)
            if not self.gcn and dst in neighbors:
                neighbors[dst].append(src)
        return neighbors

# GraphSAGE class
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_sample=10, gcn=False):
        super(GraphSAGE, self).__init__()
        self.num_layers = len(hidden_dims)
        self.gcn = gcn
        self.embed_dim = hidden_dims[-1]

        self.aggregators = nn.ModuleList()
        self.encoders = nn.ModuleList()

        # Initial layer
        aggregator = MeanAggregator(num_sample)
        encoder = Encoder(input_dim, hidden_dims[0], aggregator, num_sample, gcn)
        self.aggregators.append(aggregator)
        self.encoders.append(encoder)

        # Subsequent layers
        for i in range(1, self.num_layers):
            aggregator = MeanAggregator(num_sample)
            encoder = Encoder(hidden_dims[i - 1], hidden_dims[i], aggregator, num_sample, gcn)
            self.aggregators.append(aggregator)
            self.encoders.append(encoder)

    def forward(self, nodes, features, edge_index):
        for i in range(self.num_layers):
            features = self.encoders[i](nodes, features, edge_index)
        return features

class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, graphsage, readout="sum"):
        super(SupervisedGraphSage, self).__init__()
        self.graphsage = graphsage

        # Readout function for aggregating node embeddings into graph-level embeddings
        if readout == "mean":
            self.readout = global_mean_pool
        elif readout == "max":
            self.readout = global_max_pool
        elif readout == "sum":
            self.readout = global_add_pool
        else:
            raise ValueError("Invalid readout type.")

        # Linear classifier for graph-level embeddings
        self.graph_classifier = nn.Linear(graphsage.embed_dim, num_classes)
        nn.init.xavier_uniform_(self.graph_classifier.weight)

    def forward(self, features, edge_index, batch):
        # Get node embeddings from GraphSAGE
        nodes = torch.arange(features.size(0)).to(features.device)
        embeds = self.graphsage(nodes, features, edge_index)

        # Aggregate node embeddings into graph-level embeddings
        graph_embeds = self.readout(embeds, batch)

        # Classify the graph embeddings
        scores = self.graph_classifier(graph_embeds)
        return scores, graph_embeds

    def snnl_loss(self, graph_embeds, group_labels, T):
        # Calculate cosine similarity matrix between all graph embeddings
        similarity_matrix = F.cosine_similarity(graph_embeds.unsqueeze(1), graph_embeds.unsqueeze(0), dim=-1)
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        # Apply temperature scaling
        scaled_distance = distance_matrix / T
        scaled_distance = torch.clamp(scaled_distance, min=-50, max=50)
        weights = torch.exp(-scaled_distance)
          # Create a mask for same-group graphs (same group label)
        same_group_mask = group_labels.unsqueeze(1) == group_labels.unsqueeze(0)
        # Numerator: Sum weights where graphs are in the same group
        numerator = (weights * same_group_mask.float()).sum(dim=1)
        denominator = weights.sum(dim=1)
        # Avoid division by zero
        epsilon = 1e-8
        snnl = -torch.log((numerator + epsilon) / (denominator + epsilon))
        return snnl.mean()

    def combined_loss_function(self, original_loss, graph_embeds, group_labels, T):
        # Compute SNNL at the graph level
        snnl = self.snnl_loss(graph_embeds, group_labels, T)
        # Combine cross-entropy loss with SNNL
        combined_loss = original_loss - snnl
        return combined_loss

    def loss(self, features, edge_index, batch, labels, group_labels, T=0.1):
        # Forward pass to get predictions and graph embeddings
        scores, graph_embeds = self.forward(features, edge_index, batch)

        # Cross-Entropy Loss
        labels = labels.long()
        loss_fn = nn.CrossEntropyLoss()
        classification_loss = loss_fn(scores, labels)

        # Combine the original loss with SNNL
        combined_loss = self.combined_loss_function(classification_loss, graph_embeds, group_labels, T)
        return combined_loss

# Training function
def train_graphsage(graphsage, normal_data_list, watermark_data_list, batch_size=32, num_epochs=100, lr=0.0005, T=0.1, start_factor =1.0, end_factor=0.1):
    optimizer = torch.optim.Adam(graphsage.parameters(), lr=lr, weight_decay=5e-4)

    # Combine both normal and watermark data into one list
    combined_graph_list = normal_data_list + watermark_data_list

    # Assign group labels to each graph
    for graph in normal_data_list:
        graph.group_label = torch.tensor(0, dtype=torch.long)
    for graph in watermark_data_list:
        graph.group_label = torch.tensor(1, dtype=torch.long)

    # Create a DataLoader for batching
    data_loader = DataLoader(combined_graph_list, batch_size=batch_size, shuffle=True)
    # set up LinearLR
    total_steps = num_epochs * len(data_loader)    
    scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_steps)

    for epoch in range(num_epochs):
        graphsage.train()  # Set the model in training mode
        total_loss = 0

        # Iterate over batches of graphs
        for batch in data_loader:
            features = batch.x
            edge_index = batch.edge_index
            labels = batch.y
            batch_graph_indices = batch.batch
            group_labels = batch.group_label

            features = features.to(graphsage.graph_classifier.weight.device)
            edge_index = edge_index.to(graphsage.graph_classifier.weight.device)
            labels = labels.to(graphsage.graph_classifier.weight.device)
            batch_graph_indices = batch_graph_indices.to(graphsage.graph_classifier.weight.device)
            group_labels = group_labels.to(graphsage.graph_classifier.weight.device)

            optimizer.zero_grad()

            # Compute combined loss
            loss = graphsage.loss(features, edge_index, batch_graph_indices, labels, group_labels, T)

            loss.backward()
            optimizer.step()
             #learning rate LinearLR model
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate_graphsage(graphsage, eval_data_list, batch_size=32):
    graphsage.eval()
    eval_loader = DataLoader(eval_data_list, batch_size=batch_size, shuffle=False)

    total_loss = 0
    correct_predictions = 0
    total_graphs = 0

    with torch.no_grad():
        for batch in eval_loader:
            features = batch.x
            edge_index = batch.edge_index
            labels = batch.y
            batch_graph_indices = batch.batch

            features = features.to(graphsage.graph_classifier.weight.device)
            edge_index = edge_index.to(graphsage.graph_classifier.weight.device)
            labels = labels.to(graphsage.graph_classifier.weight.device)
            batch_graph_indices = batch_graph_indices.to(graphsage.graph_classifier.weight.device)

            # Forward pass
            scores, _ = graphsage(features, edge_index, batch_graph_indices)

            # Compute Cross-Entropy Loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(scores, labels)
            total_loss += loss.item()

            # Compute accuracy
            predicted_labels = scores.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_graphs += labels.size(0)

    avg_loss = total_loss / len(eval_loader)
    accuracy = correct_predictions / total_graphs
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy
