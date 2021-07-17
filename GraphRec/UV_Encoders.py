import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import dgl


class UV_Encoder(nn.Module):

    def __init__(self, features, embed_dim, aggregator, cuda, uv=True):
        super(UV_Encoder, self).__init__()

        self.features = features
        self.uv = uv
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes, uv_data):
        g, ratings, row_idxs, col_idxs = uv_data
        neigh_feats = self.aggregator.forward(
            nodes, g, row_idxs, col_idxs, ratings)  # user-item network

        self_feats = self.features.weight[nodes]
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear(combined))
        return combined
