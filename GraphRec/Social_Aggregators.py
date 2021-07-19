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

    def __init__(self, embed_dim, cuda):
        super(Social_Aggregator, self).__init__()

        self.device = cuda
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, uu_g, embed_u, rep):
        attention = self.att(embed_u, rep)
        attention = edge_softmax(uu_g, attention)
        uu_g.edata['h'] = embed_u * attention
        uu_g.update_all(message_func=fn.copy_edge(edge='h', out='m'),
                        reduce_func=fn.sum(msg='m', out='n'))
        feat = uu_g.ndata['n'][nodes]
        return feat
