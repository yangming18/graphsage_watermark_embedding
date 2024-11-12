# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
from torch_geometric.data import Data

# MeanAggregator class with neighbor sampling
class MeanAggregator(nn.Module):
    def __init__(self, num_sample=10):
        super(MeanAggregator, self).__init__()
        self.num_sample = num_sample

    def forward(self, nodes, adj_lists, features):
        """
        Computes the mean of sampled neighbor features for each node.

        Args:
            nodes (list or Tensor): List of node indices.
            adj_lists (dict): Dictionary mapping node indices to lists of neighbor indices.
            features (Tensor): Node feature matrix of shape (num_nodes, feature_dim).

        Returns:
            Tensor: Aggregated neighbor features of shape (len(nodes), feature_dim).
        """
        agg_feats = []
        for node in nodes:
            neighbor_ids = adj_lists[node.item()]
            if len(neighbor_ids) == 0:
                neighbor_feats = features[node].unsqueeze(0)
            else:
                # Neighbor sampling
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
        """
        Get the neighbors of the given nodes using edge_index.

        Args:
            nodes (list or Tensor): List of node indices.
            edge_index (Tensor): Edge index tensor of shape [2, num_edges].

        Returns:
            dict: Dictionary where each node maps to a list of its neighbors.
        """
        neighbors = {node.item(): [] for node in nodes}
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        edge_dict = {}
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

# SupervisedGraphSage class
class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, graphsage, readout="sum"):
        super(SupervisedGraphSage, self).__init__()
        self.graphsage = graphsage

        if readout == "mean":
            self.readout = self.mean_readout
        elif readout == "max":
            self.readout = self.max_readout
        elif readout == "sum":
            self.readout = self.sum_readout
        else:
            raise ValueError("Invalid readout type.")

        self.graph_classifier = nn.Linear(graphsage.embed_dim, num_classes)
        init.xavier_uniform_(self.graph_classifier.weight)

    def forward(self, features, edge_index):
        """
        Forward pass for graph classification.

        Args:
            features (Tensor): Node features of shape (num_nodes, feature_dim).
            edge_index (Tensor): Edge index representing the graph structure.

        Returns:
            Tensor: Scores for each class.
        """
        nodes = torch.arange(features.size(0))
        embeds = self.graphsage(nodes, features, edge_index)
        graph_embed = self.readout(embeds).unsqueeze(0)
        scores = self.graph_classifier(graph_embed)
        return scores

    def sum_readout(self, node_embeds):
        """Aggregate node embeddings by taking the sum."""
        return torch.sum(node_embeds, dim=0)

    def mean_readout(self, node_embeds):
        """Aggregate node embeddings by taking the mean."""
        return torch.mean(node_embeds, dim=0)

    def max_readout(self, node_embeds):
        """Aggregate node embeddings by taking the max."""
        return torch.max(node_embeds, dim=0)[0]  # Max pooling

    def loss(self, features, edge_index, label):
        
        scores = self.forward(features, edge_index)
        label = label.long().view(1)
        #print('scores shape:', scores.shape)
        #print('scores dtype:', scores.shape)
        #print('label shape:', label.shape)
        #print('label dtype:', label.dtype)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)
        return loss_fn(scores, label)


# Training function
def train_graphsage(graphsage, graph_list, num_epochs=100, lr=0.0005):
    """
    Train GraphSAGE model using the provided graphs and parameters.

    Args:
        graphsage (SupervisedGraphSage): The model to train.
        graph_list (list): List of graphs (Data objects) for training.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    optimizer = torch.optim.Adam(graphsage.parameters(), lr=lr, weight_decay=5e-4)


    for epoch in range(num_epochs):
        graphsage.train()
        total_loss = 0

        for graph in graph_list:
            features = graph.x  # Node features
            edge_index = graph.edge_index  # Edge index
            label = graph.y  # Graph label

            # Forward pass and loss computation
            optimizer.zero_grad()
            loss = graphsage.loss(features, edge_index, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(graph_list)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation function
def evaluate_graphsage(graphsage, graph_list):
    graphsage.eval()
    correct = 0
    total = len(graph_list)

    with torch.no_grad():
        for graph in graph_list:
            features = graph.x  # Node features
            edge_index = graph.edge_index  # Edge index
            label = graph.y.long()  # Graph label as LongTensor

            # Get prediction from the model
            scores = graphsage(features, edge_index)  # Shape: [1, num_classes]
            predicted_label = scores.argmax(dim=1).item()

            if predicted_label == label.item():
                correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

