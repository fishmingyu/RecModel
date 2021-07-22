import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import dgl.function as fn
import dgl.ops as ops
from torch.autograd.profiler import profile


class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias=True)
        self.W2 = nn.Linear(in_size, out_size, bias=True)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        # norm
        self.norm_dict = norm_dict

    def forward(self, g, multi_g, feat_dict):

        messagetime = 0
        updatetime = 0
        relunormtime = 0
        funcs = {}  # message and reduce functions dict
        # for each type of edges, compute messages and reduce them all
        messagetimeall = 0
        for srctype, etype, dsttype in g.canonical_etypes:  # "tags"
            if srctype == dsttype:  # for self loops
                start = time.time()
                messages = self.W1(feat_dict[srctype])
                end = time.time()
                messagetimeall = messagetimeall + end - start
                g.nodes[srctype].data[etype] = messages  # store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'),
                                                    fn.sum('m', 'h'))  # define message and reduce functions
            else:
                # src, dst [n_train]
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                # norm, [n_train, 1]
                norm = self.norm_dict[(srctype, etype, dsttype)]
                # feat_dict [n_users, embed_size/layer_size] or [n_users, embed_size/layer_size]
                # messages [n_train, embed_size/layer_size]
                
                start = time.time()    
                messages = norm * (self.W1(ops.copy_u(multi_g[etype], feat_dict[srctype])) + self.W2(ops.u_mul_v(multi_g[etype], feat_dict[srctype], feat_dict[dsttype])))  # compute messages
                end = time.time()
                messagetimeall = messagetimeall + end - start
                g.edges[(srctype, etype, dsttype)
                        ].data[etype] = messages  # store in edata
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'),
                                                    fn.sum('m', 'h'))  # define message and reduce functions
        # print("messagetime : %.4f " % (messagetimeall))
        messagetime = messagetimeall
        # update all, reduce by first type-wisely then across different types
        
        start = time.time()
        g.multi_update_all(funcs, 'sum') #SPMM
        end = time.time()
        # print("multi_update_all : %.4f " % (end - start))
        updatetime = end - start
        feature_dict = {}
        
        start = time.time()
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h'])  # leaky relu
            h = self.dropout(h)  # dropout
            h = F.normalize(h, dim=1, p=2)  # l2 normalize
            feature_dict[ntype] = h
        
        end = time.time()
        # print("relu&dropout&norm : %.4f " % (end - start))
        relunormtime = end - start
        return feature_dict, messagetime, updatetime, relunormtime


class NGCF(nn.Module):
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5):
        super(NGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(dst, etype=(
                srctype, etype, dsttype)).float()  # obtain degrees
            src_degree = g.out_degrees(
                src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1)  # compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()
        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers-1):
            self.layers.append(
                NGCFLayer(layer_size[i], layer_size[i+1],
                          self.norm_dict, dropout[i+1])
            )
        self.initializer = nn.init.xavier_uniform_

        # embeddings for different types of nodes
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

    def create_bpr_loss(self, users, pos_items, user_idx, item_idx):
        pos_scores = (users[user_idx] * pos_items[item_idx]).sum(1)
        # pos_scores = (users * pos_items).sum(1)

        mf_loss = nn.ReLU()(pos_scores).mean()
        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items)
                       ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, g, multi_g, user_key, item_key):
        messaget = 0
        updatet = 0
        relunormt = 0
        ngcfcalt = 0
        h_dict = {ntype: self.feature_dict[ntype] for ntype in g.ntypes}
        # obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []

        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])

        for layer in self.layers:
            start = time.time()
            h_dict, messagetime, updatetime, relunormtime = layer(g, multi_g, h_dict)
            messaget = messaget + messagetime
            updatet = updatet + updatetime
            relunormt = relunormt + relunormtime
            
            end = time.time()
            ngcfcalt = ngcfcalt + end - start
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.cat(user_embeds, 1) # cat in second dimension # layer_size_1 + layer_size_2 + ......
        item_embd = torch.cat(item_embeds, 1)
        return user_embd, item_embd, \
            messaget, updatet, relunormt, ngcfcalt

    def test(self, g, multi_g, user_key, item_key, users, pos_items):

        h_dict = {ntype: self.feature_dict[ntype] for ntype in g.ntypes}
        # obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])

        for layer in self.layers:
            h_dict, messagetime, updatetime, relunormtime = layer(g, multi_g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.cat(user_embeds, 1) # cat in second dimension # layer_size_1 + layer_size_2 + ......
        item_embd = torch.cat(item_embeds, 1)

        u_g_embeddings = user_embd
        pos_i_g_embeddings = item_embd

        return u_g_embeddings, pos_i_g_embeddings
