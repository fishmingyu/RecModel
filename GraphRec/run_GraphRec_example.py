import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from GraphRec import GraphRec
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from torch.autograd.profiler import profile
import datetime
import argparse
import utils
import os
import time

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""
def train(model, device, train_data, optimizer, scheduler, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    nodes_u, nodes_v, data_uv, data_vu, labels_list, data_uu = train_data
    optimizer.zero_grad()
    with profile(use_cuda=True,record_shapes=True) as prof:
        loss = model.loss(nodes_u, nodes_v, data_uv, data_vu, labels_list, data_uu)
        loss.backward(retain_graph=True)
        optimizer.step()
    f = open('./profile.txt', 'w')
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), file=f)
    f.close()
    scheduler.step()
    print("lr %f" % (optimizer.param_groups[0]['lr']))
    running_loss += loss.item()
    print('[epoch %d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
        epoch, running_loss, best_rmse, best_mae))
    running_loss = 0.0
    return 0


def test(model, device, test_data):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        nodes_u, nodes_v, data_uv, data_vu, tmp_target, data_uu = test_data
        val_output = model.forward(nodes_u, nodes_v, data_uv, data_vu, data_uu)
        tmp_pred.append(list(val_output.data.cpu().numpy()))
        target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=25600,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64,
                        metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.1,
                        metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=25600,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100,
                        metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/toy_dataset'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)
    
    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)
    
    # please add the validation set
    
    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    train_u = torch.LongTensor(train_u).to(device)
    train_v = torch.LongTensor(train_v).to(device)
    train_r = torch.FloatTensor(train_r).to(device)
    test_u = torch.LongTensor(test_u).to(device)
    test_v = torch.LongTensor(test_v).to(device)
    test_r = torch.FloatTensor(test_r).to(device)
    

    print("building train graph ...")
    train_uu = utils.build_uu_graph(train_u, social_adj_lists, device)
    train_uv = utils.build_uv_graph(train_u, history_u_lists, history_ur_lists, device)
    train_vu = utils.build_uv_graph(train_v, history_v_lists, history_vr_lists, device)
    
    print("building test graph ...")
    test_uu = utils.build_uu_graph(test_u, social_adj_lists, device)
    test_uv = utils.build_uv_graph(test_u, history_u_lists, history_ur_lists, device)
    test_vu = utils.build_uv_graph(test_v, history_v_lists, history_vr_lists, device)

    train_data = train_u, train_v, train_uv, train_vu, train_r, train_uu
    test_data = test_u, test_v, test_uv, test_vu, test_r, test_uu

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(
        v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(embed_dim, agg_u_social, base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(
        v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, agg_v_history, cuda=device, uv=False)

    # model
    graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
    optimizer = torch.optim.Adam(
        graphrec.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    print("start training...")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(graphrec, device, train_data,
            optimizer, scheduler, epoch, best_rmse, best_mae)
        end = time.time()
        print("train time used one epoch %.3f" % (end - start))
        expected_rmse, mae = test(graphrec, device, test_data)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
        print("rmse %.4f, mae %.4f " % (expected_rmse, mae))

if __name__ == "__main__":
    main()