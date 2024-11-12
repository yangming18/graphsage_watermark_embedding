import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, aggregator, num_sample=10, gcn=False, cuda=False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.gcn = gcn

        if self.gcn:
            self.linear = nn.Linear(input_dim, embed_dim)
        else:
            self.linear = nn.Linear(input_dim + embed_dim, embed_dim)

        if cuda:
            self.cuda()

    def forward(self, nodes, features, edge_index):
        """
        Generates embeddings for a batch of nodes.
        """
        node_adj_lists = self.get_neighbors(nodes, edge_index)
        neigh_feats = self.aggregator.forward(nodes, node_adj_lists, features)

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
        """
        neighbors = {}
        for node in nodes:
            neighbors[node] = edge_index[1][edge_index[0] == node].tolist()
        return neighbors
