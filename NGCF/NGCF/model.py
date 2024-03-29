import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.ops as ops
from memory_profiler import profile
# from torch.autograd.profiler import profile


class NGCFLayer_ori(nn.Module):
    fp = open("mprof/report0.log", "w")
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer_ori, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict
    @profile(stream = fp)
    def forward(self, g, feat_dict):
        funcs = {} #message and reduce functions dict
        #for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype: #for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages   #store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (self.W1(feat_dict[srctype][src]) + self.W2(feat_dict[srctype][src]*feat_dict[dsttype][dst])) #compute messages
                g.edges[(srctype, etype, dsttype)].data[etype] = messages  #store in edata
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions

        g.multi_update_all(funcs, 'sum') #update all, reduce by first type-wisely then across different types
        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCFLayer_our1(nn.Module):
    fp = open("mprof/report1.log", "w")
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer_our1, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict
    @profile(stream = fp)
    def forward(self, g, feat_dict):
        funcs = {} #message and reduce functions dict
        #for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype: #for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages   #store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
            else:
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (self.W1(ops.copy_u(g.edge_type_subgraph([etype]), feat_dict[srctype])) + \
                 self.W2(ops.u_mul_v(g.edge_type_subgraph([etype]), feat_dict[srctype], feat_dict[dsttype])))  # compute messages
                g.edges[(srctype, etype, dsttype)].data[etype] = messages  #store in edata
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions

        g.multi_update_all(funcs, 'sum') #update all, reduce by first type-wisely then across different types
        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCFLayer_our2(nn.Module):
    fp = open("mprof/report2.log", "w")
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer_our2, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict
    @profile(stream = fp)
    def forward(self, g, feat_dict):
        funcs = {} #message and reduce functions dict
        #for each type of edges, compute messages and reduce them all
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype: #for self loops
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages   #store in ndata
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions
            else:
                
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (ops.copy_u(g.edge_type_subgraph([etype]), self.W1(feat_dict[srctype])) + \
                        self.W2(ops.u_mul_v(g.edge_type_subgraph([etype]), feat_dict[srctype], feat_dict[dsttype])))  # compute messages
                
                g.edges[(srctype, etype, dsttype)].data[etype] = messages  #store in edata
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))  #define message and reduce functions

        g.multi_update_all(funcs, 'sum') #update all, reduce by first type-wisely then across different types
        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCFLayer_our3(nn.Module):
    fp = open("mprof/report3.log", "w")
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer_our3, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict
    @profile(stream = fp)
    def forward(self, g, feat_dict):
        #for each type of edges, compute messages and reduce them all
        for ntype in g.ntypes: 
            if ntype == 'user':
                norm = self.norm_dict[('item', 'iu', 'user')]
                g.nodes[ntype].data['h'] = ops.copy_u_sum(g.edge_type_subgraph(['user_self']), self.W1(feat_dict['user'])) + \
                    ops.u_mul_e_sum(g.edge_type_subgraph(['iu']), self.W1(feat_dict['item']), norm) + \
                     ops.copy_e_sum(g.edge_type_subgraph(['iu']), norm * self.W2(ops.u_mul_v(g.edge_type_subgraph(['iu']), feat_dict['item'], feat_dict['user'])))
            else: # to item
                norm = self.norm_dict[('user', 'ui', 'item')]
                g.nodes[ntype].data['h'] = ops.copy_u_sum(g.edge_type_subgraph(['item_self']), self.W1(feat_dict['item'])) + \
                    ops.u_mul_e_sum(g.edge_type_subgraph(['ui']), self.W1(feat_dict['user']), norm) + \
                     ops.copy_e_sum(g.edge_type_subgraph(['ui']), norm * self.W2(ops.u_mul_v(g.edge_type_subgraph(['ui']), feat_dict['user'], feat_dict['item'])))

        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCFLayer_our4(nn.Module):
    fp = open("mprof/report4.log", "w")
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer_our4, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict
    @profile(stream = fp)
    def forward(self, g, feat_dict):
        #for each type of edges, compute messages and reduce them all
        for ntype in g.ntypes: 
            if ntype == 'user':
                norm = self.norm_dict[('item', 'iu', 'user')]
                norm_src = self.norm_dict['item']
                norm_dst = self.norm_dict['user']
                g.nodes[ntype].data['h'] = self.W1(ops.copy_u_sum(g.edge_type_subgraph(['user_self']), feat_dict['user']) + \
                    ops.u_mul_e_sum(g.edge_type_subgraph(['iu']), feat_dict['item'], norm)) + \
                     self.W2(ops.copy_e_sum(g.edge_type_subgraph(['iu']), ops.u_mul_v(g.edge_type_subgraph(['iu']), feat_dict['item'] * norm_src, feat_dict['user'] * norm_dst)))
            else: # to item
                norm = self.norm_dict[('user', 'ui', 'item')]
                norm_src = self.norm_dict['user']
                norm_dst = self.norm_dict['item']
                g.nodes[ntype].data['h'] = self.W1(ops.copy_u_sum(g.edge_type_subgraph(['item_self']), feat_dict['item']) + \
                    ops.u_mul_e_sum(g.edge_type_subgraph(['ui']), feat_dict['user'], norm)) + \
                     self.W2(ops.copy_e_sum(g.edge_type_subgraph(['ui']), ops.u_mul_v(g.edge_type_subgraph(['ui']), feat_dict['user'] * norm_src, feat_dict['item'] * norm_dst)))

        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCFLayer_our5(nn.Module):
    fp = open("mprof/report5.log", "w")
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer_our5, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        #weights for different types of messages
        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        #leaky relu
        self.leaky_relu = nn.LeakyReLU(0.2)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #initialization
        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        #norm
        self.norm_dict = norm_dict
    @profile(stream = fp)
    def forward(self, g, feat_dict):
        norm_iu = self.norm_dict[('item', 'iu', 'user')]
        norm_ui = self.norm_dict[('user', 'ui', 'item')]
        ui_g = g.edge_type_subgraph(['ui'])
        iu_g = g.edge_type_subgraph(['iu']) 
        norm_user = self.norm_dict['user']
        norm_item = self.norm_dict['item']
        edge_message = ops.u_mul_v(ui_g, feat_dict['user'] * norm_user, feat_dict['item'] * norm_item)
        g.nodes['user'].data['h'] = self.W1(feat_dict['user'] + \
            ops.u_mul_e_sum(iu_g, feat_dict['item'], norm_iu)) + \
                self.W2(ops.copy_e_sum(iu_g, edge_message))
        g.nodes['item'].data['h'] = self.W1(feat_dict['item'] + \
            ops.u_mul_e_sum(ui_g, feat_dict['user'], norm_ui)) + \
                self.W2(ops.copy_e_sum(ui_g, edge_message))

        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h']) #leaky relu
            h = self.dropout(h) #dropout
            h = F.normalize(h,dim=1,p=2) #l2 normalize
            feature_dict[ntype] = h
        return feature_dict

class NGCF(nn.Module):
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5, model_type=0):
        super(NGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            
            dst_degree = g.in_degrees(dst, etype=(srctype, etype, dsttype)).float() #obtain degrees
            src_degree = g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1) #compute norm
            self.norm_dict[(srctype, etype, dsttype)] = norm
        for ntype in g.ntypes: 
            nodes = g.nodes(ntype)
            if ntype == 'user':
                nodes_degree = g.in_degrees(nodes, etype=('user', 'ui', 'item')).float()
            else:
                nodes_degree = g.out_degrees(nodes, etype=('item', 'iu', 'user')).float()
            norm = torch.pow(nodes_degree, -0.5).unsqueeze(1)
            self.norm_dict[ntype] = norm

        self.layers = nn.ModuleList()
        if model_type == 0:
            NGCFLayer = NGCFLayer_ori
        elif model_type == 1:
            NGCFLayer = NGCFLayer_our1                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        elif model_type == 2:
            NGCFLayer = NGCFLayer_our2
        elif model_type == 3:
            NGCFLayer = NGCFLayer_our3
        elif model_type == 4:
            NGCFLayer = NGCFLayer_our4
        else:
            NGCFLayer = NGCFLayer_our5 

        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers-1):
            self.layers.append(
                NGCFLayer(layer_size[i], layer_size[i+1], self.norm_dict, dropout[i+1])
            )
    
        self.initializer = nn.init.xavier_uniform_

        #embeddings for different types of nodes
        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    
    # @profile
    def forward(self, g, user_key, item_key, users, pos_items, neg_items):
        h_dict = {ntype : self.feature_dict[ntype] for ntype in g.ntypes}
        #obtain features of each layer and concatenate them all
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)

        u_g_embeddings = user_embd[users, :]
        pos_i_g_embeddings = item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
