import dgl
import torch

def build_uu_graph(nodes_u, social_adj_lists, u2e, device):
    row_idxs = []
    col_idxs = torch.Tensor([]).to(device).int()
    for node in nodes_u:
        num_history = len(social_adj_lists[int(node)])
        row_idxs = row_idxs + list(social_adj_lists[int(node)])
        col_idxs = torch.cat((col_idxs, node.repeat(num_history)), dim=0)
    row_idxs = torch.Tensor(row_idxs).to(device).long()
    uu_g = dgl.graph(data=(row_idxs, col_idxs),
                         idtype=torch.int32, device=device)
    print('uu_g.nodes %d uu_g.edges %d' % (uu_g.num_nodes(), uu_g.num_edges()))
    return (uu_g, u2e.weight[row_idxs], u2e.weight[col_idxs])

def build_uv_graph(nodes_u, history_u_lists, history_r_lists, u2e, v2e, r2e, device): #uv=true
    col_idxs = []
    ratings = []
    row_idxs = torch.Tensor([]).to(device).int()
    for node in nodes_u:
        num_history = len(history_u_lists[int(node)])
        row_idxs = torch.cat((row_idxs, node.repeat(num_history)), dim=0)  #nodes_u
        col_idxs = col_idxs + history_u_lists[int(node)]
        ratings = ratings + history_r_lists[int(node)]
    col_idxs = torch.Tensor(col_idxs).to(device).long()
    ratings = torch.Tensor(ratings).to(device).long()
    graph_data = {('item', 'looked-by', 'user'):(col_idxs, row_idxs)}
    uv_g = dgl.heterograph(data_dict=graph_data, idtype=torch.int32, device=device)
    print('uv_g.nodes for users %d uv_g.nodes for items %d uv_g.edges %d' % (uv_g.num_nodes('user'), uv_g.num_nodes('item'), uv_g.num_edges()))
    return (uv_g, r2e.weight[ratings], v2e.weight[col_idxs], u2e.weight[row_idxs])

def build_vu_graph(nodes_v, history_v_lists, history_r_lists, u2e, v2e, r2e, device):
    col_idxs = []
    ratings = []
    row_idxs = torch.Tensor([]).to(device).int()
    for node in nodes_v:
        num_history = len(history_v_lists[int(node)])
        row_idxs = torch.cat((row_idxs, node.repeat(num_history)), dim=0)
        col_idxs = col_idxs + history_v_lists[int(node)] #item
        ratings = ratings + history_r_lists[int(node)]
    col_idxs = torch.Tensor(col_idxs).to(device).long()
    ratings = torch.Tensor(ratings).to(device).long()
    graph_data = {('user', 'looks', 'item'):(col_idxs, row_idxs)}
    vu_g = dgl.heterograph(data_dict=graph_data, idtype=torch.int32, device=device)
    print('vu_g.nodes for users %d vu_g.nodes for items %d vu_g.edges %d' % (vu_g.num_nodes('user'), vu_g.num_nodes('item'), vu_g.num_edges()))
    return (vu_g, r2e.weight[ratings], u2e.weight[col_idxs], v2e.weight[row_idxs])
