import dgl
import torch

def build_uu_graph(nodes_u, social_adj_lists, device):
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
    return (uu_g, row_idxs, col_idxs)

def build_uv_graph(nodes_v, history_uv_lists, history_r_lists, device):
    row_idxs = []
    ratings = []
    col_idxs = torch.Tensor([]).to(device).int()
    for node in nodes_v:
        num_history = len(history_uv_lists[int(node)])
        col_idxs = torch.cat((col_idxs, node.repeat(num_history)), dim=0)
        row_idxs = row_idxs + history_uv_lists[int(node)]
        ratings = ratings + history_r_lists[int(node)]
    row_idxs = torch.Tensor(row_idxs).to(device).long()
    ratings = torch.Tensor(ratings).to(device).long()
    uv_g = dgl.graph(data=(row_idxs + max(col_idxs), col_idxs),
                         idtype=torch.int32, device=device)
    print('uv_g.nodes %d uv_g.edges %d' % (uv_g.num_nodes(), uv_g.num_edges()))
    return (uv_g, ratings, row_idxs, col_idxs)