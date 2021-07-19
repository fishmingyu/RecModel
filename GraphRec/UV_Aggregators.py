import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention
import dgl
import dgl.ops as F
import dgl.function as fn
import time
from dgl.ops import edge_softmax


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, embed_dim, cuda, uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.device = cuda
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU()
        )

    def forward(self, nodes, uv_g, emb_uv, rep, emb_r):
        t_cat = torch.cat((emb_uv, emb_r), 1)
        ohistory = self.mlp(t_cat)
        attention = self.att(ohistory, rep)
        attention = edge_softmax(uv_g, attention)
        uv_g.edata['h'] = ohistory * attention
        if self.uv == False: #user->item
            uv_g.multi_update_all({'looks': (fn.copy_e('h', 'm'), fn.sum('m', 'n'))}, "sum")
            feat = uv_g.nodes['item'].data['n']
        else: #item->user
            uv_g.multi_update_all({'looked-by': (fn.copy_e('h', 'm'), fn.sum('m', 'n'))}, "sum")
            feat = uv_g.nodes['user'].data['n']
        return feat[nodes]
