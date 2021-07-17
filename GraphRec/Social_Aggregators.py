import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention
import dgl
import dgl.ops as F
import dgl.function as fn
from dgl.ops import edge_softmax


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, u2e, embed_dim, cuda):
        super(Social_Aggregator, self).__init__()

        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, uu_g, row_idxs, col_idxs):
        embed_u = self.u2e.weight[row_idxs]
        rep = self.u2e.weight[col_idxs]
        attention = self.att(embed_u, rep)
        attention = edge_softmax(uu_g, attention)
        uu_g.edata['h'] = embed_u * attention
        uu_g.update_all(message_func=fn.copy_edge(edge='h', out='m'),
                        reduce_func=fn.sum(msg='m', out='n'))
        feat = uu_g.ndata['n'][nodes]
        return feat
