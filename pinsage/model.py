import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
import tqdm
import time
from torch.autograd.profiler import profile

import layers
import sampler as sampler_module
import evaluation

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector( # get the initial feature of nodes
            full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers) # used to get the final embedding
        self.scorer = layers.ItemToItemScorer(full_graph, ntype) # given the embedding of head & tail nodes, compute the scores on the edges.

    def forward(self, pos_graph, neg_graph, blocks): # pos_graph, neg_graph, and the last layer of blocks: corresponds to the nodes (heads+tails+neg_tails) in each batch
        h_item = self.get_repr(blocks) # get final embedding of (heads + tails + neg_tails) in one batch
        pos_score = self.scorer(pos_graph, h_item) # scores on the edges (heads -> tails)
        neg_score = self.scorer(neg_graph, h_item) # scores on the edges (heads -> neg_tails)
        return (neg_score - pos_score + 1).clamp(min=0) # margin hinge loss. Here margin = 1. # of edges on pos_graph and neg_graph are equal. 

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata) # project the original feature of input nodes to a vector with length of hidden_dims
        h_item_dst = self.proj(blocks[-1].dstdata) # project the original feature of output nodes to a vector with length of hidden_dims
        return h_item_dst + self.sage(blocks, h_item) # DGL specific implementation: adding the GNN output with the node's own learnable embedding as the final item representation instead. (Like ResNet?)


def train(dataset, args):
    g = dataset['train-graph']
    print("%d nodes and %d edges of graph" % (g.num_nodes(), g.num_edges()))
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(
        g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(
        g.number_of_nodes(item_ntype))

    # Prepare torchtext dataset and vocabulary
    fields = {}
    examples = []
    for key, texts in item_texts.items():
        fields[key] = torchtext.legacy.data.Field(
            include_lengths=True, lower=True, batch_first=True)
    for i in range(g.number_of_nodes(item_ntype)):
        example = torchtext.legacy.data.Example.fromlist(
            [item_texts[key][i] for key in item_texts.keys()],
            [(key, fields[key]) for key in item_texts.keys()])
        examples.append(example)
    textset = torchtext.legacy.data.Dataset(examples, fields)
    for key, field in fields.items():
        field.build_vocab(getattr(textset, key))
        #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')

    # Sampler
    # a batch contains (heads, tails, neg_tails)
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    # construct (pos_graph, neg_graph, blocks) for training this batch, given (heads, tails, neg_tails)
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)

    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset)
    # when next(), return (pos_graph, neg_graph, blocks)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    # when next(), return blocks
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)
    # dataset_iter = iter(dataset)
    # for indices in batch_sampler:
    #   yield collate_fn([next(data_iter) for _ in indices])

    # Model
    model = PinSAGEModel(g, item_ntype, textset,
                         args.hidden_dims, args.num_layers).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    
    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        
        for batch_id in tqdm.trange(args.batches_per_epoch):
            
            # startsp = time.time()
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # endsp = time.time()
            # print("sample time %.3f" % (endsp - startsp))
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            # 
            # with profile(use_cuda=False,record_shapes=True) as prof:
            loss = model(pos_graph, neg_graph, blocks).mean()
            # f = open('./profile.txt', 'w')
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=30), file=f)
            # f.close()
        # Evaluate
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.number_of_nodes(
                item_ntype)).split(args.batch_size)
            h_item_batches = []
            for blocks in dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)

            print(evaluation.evaluate_nn(
                dataset, h_item, args.k, args.batch_size))


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32) # 3706 item nodes, 9746 total nodes
    # can also be "cuda:0"
    parser.add_argument('--device', type=str, default='cpu') # 'cuda:0'
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batches-per-epoch', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    train(dataset, args)