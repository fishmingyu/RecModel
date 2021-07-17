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

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda, uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU()
        )

    def forward(self, nodes, uv_g, row_idxs, col_idxs, ratings):
        if self.uv == True:
            emb_uv = self.v2e.weight[row_idxs]
            rep = self.u2e.weight[col_idxs]
        else:
            emb_uv = self.u2e.weight[row_idxs]
            rep = self.v2e.weight[col_idxs]
        emb_r = self.r2e.weight[ratings]
        t_cat = torch.cat((emb_uv, emb_r), 1)
        ohistory = self.mlp(t_cat)
        attention = self.att(ohistory, rep)
        torch.cuda.synchronize()
        start = time.time()
        attention = edge_softmax(uv_g, attention)
        uv_g.edata['h'] = ohistory * attention
        uv_g.update_all(message_func=fn.copy_edge(edge='h', out='m'),
                        reduce_func=fn.sum(msg='m', out='n'))
        torch.cuda.synchronize()
        end = time.time()
        print("end - start: %d", end - start)
        feat = uv_g.ndata['n'][nodes]
        return feat
