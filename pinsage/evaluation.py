import numpy as np
import torch
import pickle
import dgl
import argparse

def prec(recommendations, ground_truth): # ground_truth = val_matrix
    n_users, n_items = ground_truth.shape # 6040, 3706
    # print("n_users: ", n_users, "n_items: ", n_items)
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K) # [0,0,...,0,1,1,...,1,2,2,...,2,...]
    item_idx = recommendations.flatten() # shape = [num_users * K]. [60400, ]
    # print("item_idx shape: ", item_idx.shape)
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K)) # element: hit = 1; miss = 0
    hit = relevance.any(axis=1).mean() # if there exists the item in val_matrix within theses K predictions, it would be a hit, otherwise a miss. 
    return hit

class LatestNNRecommender(object):
    def __init__(self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        n_users = full_graph.number_of_nodes(self.user_ntype)
        latest_interactions = dgl.sampling.select_topk(graph_slice, 1, self.timestamp, edge_dir='out') # choose one edge (user->item) with the largest timestamp (latest) for one node (user). Return a DGLGraph (induced by these edges)
        user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst') # get the edges
        # each user should have at least one "latest" interaction
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(device=h_item.device)
            dist = h_item[latest_item_batch] @ h_item.t() # dot product. [batch_size, hidden_dim] * [hidden_dim, num_items]. 
            # print('dist shape: ', dist.shape) # Dist shape: [batch_size, num_items]
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf # if item is connected to the user, set distance small (won't choose interacted_items)
            recommended_batches.append(dist.topk(K, 1)[1]) # the k largest distance within the num_items item for the latest_item_batch

        recommendations = torch.cat(recommended_batches, 0)
        # print('recommendations shape: ', recommendations.shape) # [num_users, K]. MovieLens with K = 10: [6040, 10]
        return recommendations


def evaluate_nn(dataset, h_item, k, batch_size):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size)

    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy() # given a user. 1. find the latest item (largest timestamp), 2. find top K nearest items to this "latest item" (which is not connected to the user)
    return prec(recommendations, val_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, 'rb') as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
