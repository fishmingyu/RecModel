import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader
import sys

def compact_and_copy(frontier, seeds):
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items(): # don't know the usage of thie for loop. 
        if col == dgl.EID: # Seems that col == dgl.EID == '_ID'
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0] # 'watched'
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0] # 'watched-by'
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,)) # sample `batch_size` item nodes. 
            sample = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype]) # item -> user -> item
                # return a two-element list. sample[0]: a tensor with shape: [batch_size, 3]. Each row is a trace with a starting nodes 'head'. 
                # sample[1]: shape: [3], shows the node type ID of the trace. In this example, would be [0, 1, 0]
            tails = sample[0][:, 2] # positive examples of heads (heads: queries, tails: positive items), sampled by random-walk
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,)) # # just randomly sample `batch_size` item nodes again. Used as negative items.`
            mask = (tails != -1) # if it cannot find the edge with regulated e_type, the trace will return `-1`. The mask is to avoid these traces. For the normal trace, it should be True. 
            yield heads[mask], tails[mask], neg_tails[mask]

class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length, # given input seeds, have multiple times random-walk, then count the visit numbers for the neighbors of the input seeds, choose the top K neighbour nodes. 
                random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None): # generate blocks for each GNN layer, given seeds (the top layer samples)
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds) # PinSAGE sampler
            if heads is not None: # when training
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2] # eid for pos edges (heads->tails) and neg edges (heads->neg_tails)
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids) # to avoid 'info leakage' during training, remove positive edges (heads->tails) and negative edges (heads->neg_tails)
                    #print(old_frontier)
                    #print(frontier)
                    #print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
            block = compact_and_copy(frontier, seeds) # generate block given frontier and seeds
            seeds = block.srcdata[dgl.NID] # get new seeds for the lower layer. 
            blocks.insert(0, block) # List insert()
        return blocks # blocks[0] for layer 0, blocks[1] for layer 1, ...

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID] # seeds = heads + tails + neg_tails

        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]

def assign_textual_node_features(ndata, textset, ntype):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    block : DGLHeteroGraph
        First element of the compacted blocks, with "dgl.NID" as the
        corresponding node ID in the original graph, hence the index to the
        text dataset.

        The numericalized tokens (and lengths if available) would be stored
        onto the blocks as new node features.
    textset : torchtext.data.Dataset
        A torchtext dataset whose number of examples is the same as that
        of nodes in the original graph.
    """
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.fields.items():
        examples = [getattr(textset[i], field_name) for i in node_ids]

        tokens, lengths = field.process(examples)

        if not field.batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + '__len'] = lengths

def assign_features_to_blocks(blocks, g, textset, ntype):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)

class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, textset):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        self.textset = textset

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples): # samples shape: [batch_size]. Contains a batch from the test dataset (torch.arange(g.number_of_nodes(item_ntype))
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch) # seeds = batch
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        return blocks
