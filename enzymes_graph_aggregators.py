import torch.nn as nn
import torch
import random  

class MeanAggregator(nn.Module):
    def __init__(self, num_sample=10):
        super(MeanAggregator, self).__init__()
        self.num_sample = num_sample

    def forward(self, nodes, adj_lists, features):
        """
        Computes the mean of sampled neighbor features for each node.

        Args:
            nodes (list): List of node indices.
            adj_lists (dict): Dictionary mapping node indices to lists of neighbor indices.
            features (Tensor): Node feature matrix of shape (num_nodes, feature_dim).

        Returns:
            Tensor: Aggregated neighbor features of shape (len(nodes), feature_dim).
        """
        agg_feats = []
        for node in nodes:
            neighbor_ids = adj_lists[node]
            if len(neighbor_ids) == 0:
                # If a node has no neighbors, use its own features
                neighbor_feats = features[node].unsqueeze(0)
            else:
                # Sample neighbors if necessary
                if len(neighbor_ids) > self.num_sample:
                    sampled_neighbors = random.sample(neighbor_ids, self.num_sample)
                else:
                    sampled_neighbors = neighbor_ids
                neighbor_feats = features[sampled_neighbors]
            agg_feats.append(neighbor_feats.mean(dim=0))
        return torch.stack(agg_feats)
