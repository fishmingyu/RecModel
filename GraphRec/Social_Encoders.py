import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import dgl


class Social_Encoder(nn.Module):

    def __init__(self, embed_dim, aggregator, base_model, cuda):
        super(Social_Encoder, self).__init__()

        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes, data_uu, data_uv):
        g, emb_u, rep = data_uu
        neigh_feats = self.aggregator.forward(nodes, g, emb_u, rep)  # user-user network
        self_feats = self.base_model(nodes, data_uv)
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear(combined))

        return combined
