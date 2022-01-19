from audioop import ulaw2lin
from re import L
import pandas as pd
import numpy as np
import random as rd
import torch
import dgl

def to_etype_name(rating):
    return str(rating).replace('.', '_')

class DataLoader(object):
    def __init__(self, file_path, batch_size, test_frac, valid_frac, model_type):
        """
        initiate the dataloader and process raw data to a hetero graph
        file_path: the path of input file
        batch_size: batch of nodes while training
        [TODO] model_type: input model(NGCF, GCMC, etc.)
        """
        self.file_path = file_path
        self.test_frac = test_frac
        self.valid_frac = valid_frac
        self.colname = ['user', 'item', 'rating']
        self.model_type = model_type
        self.batch_size = batch_size
        self.process()

    def readFile(self):
        """
        input: coo file(csv/txt)
        attribute: user, item, (rating), (timestamp)
        return user item pair list
        """
        format = self.file_path.split(sep='.')[-1]
        if format == 'csv':
            ui_list = pd.read_csv(self.file_path, sep=',')
        elif format == 'txt':
            txt = np.loadtxt(self.file_path)
            ui_list = pd.DataFrame(txt)
        self.attributes = list(ui_list)
        attr_size = len(self.attributes)
        if attr_size == 2:
            ui_list.columns = ['user', 'item']
            self.rating_flag = False
        elif attr_size >= 3:
            for i in range(attr_size):
                ui_list.rename(columns = {self.attributes[i]:self.colname[i]})
                if i == 3:
                    break
            self.rating_flag = True
        return ui_list

    def train_test_split(self, ui_list):
        """
        Shuffle csv data and split it to test and train
        """
        pairs = len(ui_list)
        shuffled_idx = np.random.permutation(pairs) 
        self.num_test = round(self.test_frac * len(ui_list))
        all_train_list = ui_list.iloc[shuffled_idx]
        self.num_valid = round(self.valid_frac * len(all_train_list))
        self.users = np.unique(all_train_list['user'].values.astype(int))
        self.items = np.unique(all_train_list['item'].values.astype(int))
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        print(self.n_users)
        tmp_train_list = all_train_list.iloc[self.num_test :]
        self.test_list = all_train_list.iloc[: self.num_test]
        if self.num_valid == 0:
            self.train_list = tmp_train_list
        else:
            self.valid_list = tmp_train_list.iloc[: self.num_valid]
            self.train_list = tmp_train_list.iloc[self.num_valid: ]
        # self.possible_ratings = np.unique(self["rating"].values)

    def ui_pair(self, list_info):
        """
        pack the user item pairs with ratings
        """
        rating_pairs = (list_info["user"].values.astype(np.int64), list_info["item"].values.astype(np.int64))
        if self.rating_flag:
            rating_values = list_info["rating"].values.astype(np.float32)
            return rating_pairs, rating_values
        else:
            return rating_pairs
    
    def gen_enc_graph(self, rating_pairs, rating_values):
        """
        generate graph with different ratings
        """
        data_dict = dict()
        num_nodes_dict = {'user': self.n_users, 'movie': self.n_items}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_ratings:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        return graph

    def gen_graph(self, rating_pairs):
        """
        generate heterograph"""
        user_item_src, user_item_dst = rating_pairs
        data_dict = {
                ('user', 'ui', 'item') : (user_item_src, user_item_dst),
                ('item', 'iu', 'user') : (user_item_dst, user_item_src)
                }
        num_dict = {
            'user': self.n_users, 'item': self.n_items
        }

        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)

    
    def get_batch(self):
        """
        sample one pair batch from a graph
        """
        if self.batch_size <= self.n_users:
            users = rd.sample(list(self.users), self.batch_size)
        else:
            users = [rd.choice(list(self.users)) for _ in range(self.batch_size)]
        users = torch.LongTensor(users)
        # sub_g = self.g.edge_type_subgraph(['ui'])
        def check_users(users):
            users_with_item = []
            for u in users:
                edges = self.g.out_edges(u, etype='ui')
                items = edges[1]
                if len(items) == 0:
                    continue
                users_with_item.append(u)
            return users_with_item

        def sample_items_for_u(u):
            # sample num pos items for u-th user
            pos_edges = self.g.out_edges(u, etype='ui')
            pos_items = pos_edges[1]
            n_pos_items = len(pos_items)

            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]
            u = [u.tolist()]
            while True:
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in pos_items:
                    neg_i_id = neg_id
                    break
            return pos_i_id, neg_i_id

        pos_items, neg_items = [], []
        users_with_item = check_users(users)
        for u in users_with_item:
            pos_item, neg_item = sample_items_for_u(u)
            pos_items.append(pos_item)
            neg_items.append(neg_item)

        return users_with_item, pos_items, neg_items

    def process(self):
        """
        processing the dataset
        """
        ui_list = self.readFile()
        self.train_test_split(ui_list)
        rating_pairs = self.ui_pair(self.train_list)
        self.gen_graph(rating_pairs)
        print(self.g)

if __name__ == '__main__':
    batch_size = 32
    A = DataLoader("Data/ratings_Books.csv", batch_size, 0.1, 0.1, 'NGCF')
    users, pos_items, neg_items = A.get_batch()