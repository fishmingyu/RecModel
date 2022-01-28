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
    def __init__(self, file_path, new_file, output_path, batch_size, test_frac, valid_frac, model_type):
        """
        initiate the dataloader and process raw data to a hetero graph
        file_path: the path of input file
        new_file: whether generate a new file of reordered csv
        batch_size: batch of nodes while training
        [TODO] model_type: input model(NGCF, GCMC, etc.)
        """
        self.file_path = file_path
        self.new_file = new_file
        self.output_path = output_path
        self.test_frac = test_frac
        self.valid_frac = valid_frac
        self.colname = ['user', 'item', 'rating', 'timestamp']
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
            ui_list = pd.read_csv(self.file_path, sep=',', header=None)
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
                ui_list = ui_list.rename(columns = {self.attributes[i]:self.colname[i]})
                if i == len(self.colname) - 1:
                    break
            self.rating_flag = True
        return ui_list

    def train_test_split(self, ui_list):
        """
        Shuffle csv data and split it to test and train
        """
        self.pairs = len(ui_list)
        shuffled_idx = np.random.permutation(self.pairs) 
        self.num_test = round(self.test_frac * len(ui_list))
        
        # print(all_train_list['user'].values)
        _, idx_u = np.unique(ui_list['user'].values, return_index = True)
        _, idx_i = np.unique(ui_list['item'].values, return_index = True)

        self.users_reorder = ui_list['user'].values[np.sort(idx_u)]
        self.items_reorder = ui_list['item'].values[np.sort(idx_i)]

        self.n_users = len(self.users_reorder)
        self.n_items = len(self.items_reorder)

        self.users_map = dict(zip(self.users_reorder, range(self.n_users))) # key relabel, new label
        self.items_map = dict(zip(self.items_reorder, range(self.n_items)))

        all_train_list = ui_list.iloc[shuffled_idx]

        self.num_valid = round(self.valid_frac * len(all_train_list))
        tmp_train_list = all_train_list.iloc[self.num_test :]
        self.test_list = all_train_list.iloc[: self.num_test]
        if self.num_valid == 0:
            self.train_list = tmp_train_list
        else:
            self.valid_list = tmp_train_list.iloc[: self.num_valid]
            self.train_list = tmp_train_list.iloc[self.num_valid: ]
        # self.possible_ratings = np.unique(self["rating"].values)
        if self.new_file:
            ui_list['user'] = ui_list['user'].map(self.users_map)
            ui_list['item'] = ui_list['item'].map(self.items_map)
            ui_list.to_csv(self.output_path, index=None)


    def ui_pair(self, list_info):
        """
        pack the user item pairs with ratings
        """
        rating_row = []
        rating_col = []
        for i in range(len(list_info)):
            user = list_info["user"].values[i]
            item = list_info["item"].values[i]
            rating_row.append(self.users_map[user])
            rating_col.append(self.items_map[item])
        
        rating_pairs = (np.array(rating_row).astype(np.int64), np.array(rating_col).astype(np.int64))
        if self.rating_flag:
            rating_values = list_info["rating"].values.astype(np.float32)
            return rating_pairs, rating_values

    def gen_enc_graph(self, rating_pairs):
        """
        generate graph with different ratings
        """
        data_dict = dict()
        num_nodes_dict = {'user': self.n_users, 'movie': self.n_items}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_ratings:
            ridx = np.where(self.rating_values == rating)
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
        users, items = rating_pairs
        data_dict = {
                ('user', 'ui', 'item') : (users, items),
                ('item', 'iu', 'user') : (items, users)
                }
        num_dict = {
            'user': self.n_users, 'item': self.n_items
        }

        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)

    
    def get_batch(self):
        """
        sample one pair batch from a graph
        """
        users_src, items_dst = self.train_pairs
        if self.batch_size <= self.n_users:
            u = rd.sample(list(users_src), self.batch_size)
        else:
            u = [rd.choice(list(users_src)) for _ in range(self.batch_size)]
        u = torch.LongTensor(u)
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
        users_with_item = check_users(u)
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
        if self.rating_flag:
            rating_pairs, rating_values = self.ui_pair(self.train_list)
        else:
            rating_pairs = self.ui_pair(self.train_list)
        self.gen_graph(rating_pairs)
        self.train_pairs = rating_pairs
        print(self.g)

if __name__ == '__main__':
    batch_size = 32
    A = DataLoader(file_path="Data/ratings_Books.csv", new_file=True, output_path = "Data/output.csv", batch_size=batch_size, test_frac = 0.1, valid_frac = 0.1, model_type = 'NGCF')
    users, pos_items, neg_items = A.get_batch()