## User item based GNN recommendation system

We have selected three representative models that work in recommender systems using GNN.

### Unified dataloader

Input: A COO(Coordinate) file, default tags for columns are users, items followed by ratings and timestamp. Note that we could also process the situation that timestamp or rating is missing.

Output: A heterogeneous graph in dgl format.
We modified the following three models from [dgl/example](https://github.com/dmlc/dgl/tree/master/examples).

[TODO] 1. Fit the dataloader in more models. 2. Add parser
You could run this dataloader by the following code.

```bash
cd Data
bash load_data.sh
cd ..
python dataLoader.py
```

### GCMC

This model considers matrix completion for recommender systems from the point of view of link prediction on graphs. It first uses one-layer GCN to build a graph auto encoder, and then use bilinear decoder(SDDMM + softmax) to gain an estimated rating.

### NGCF

Based on collabortive filtering, it processes the node embedding with multi-layer of graph convolution. This model leads to the expressive modeling of high-order connectivity in user-item graph.

### pinsage

Pinsage is a classic model of RecSys with dedicated designs of sampling strategies. The building block of pinsage sources from graphsage.
