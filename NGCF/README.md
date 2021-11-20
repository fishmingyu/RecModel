# Different DGL Implementations of the NGCF Model

We modified NGCF model by using different dgl's API to reduce the memory and time usage.
The author's codes of implementation is in [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).


The graph dataset used in this example 
---------------------------------------
Gowalla: This is the check-in dataset obtained from Gowalla, where users share their locations by checking-in. To ensure the quality of the dataset, we use the 10-core setting, i.e., retaining users and items with at least ten interactions. The dataset used can be found [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data).

Statistics:
- Users: 29858
- Items: 40981
- Interactions: 1027370
- Density: 0.00084

Amazon-Book

Statistics:
- Users: 52643
- Items: 91599
- Interactions: 2931465
- Density: 0.00061

Amazon-Review: This is a ratings-only dataset obtained from Amazon, where users give feedback on the products they bought. We process the original CSV format into user_item lists.  The dataset could be found [here](http://deepyeti.ucsd.edu/jianmo/amazon/index.html). 

Statistics:
- Users: 15167257
- Items: 43531850
- Interactions: 228594421
- Density: 0.00000


How to run example files
--------------------------------
First to get the data, in the Data folder, run

```bash
sh load_gowalla.sh 
sh load_amazon-book.sh
```
optional
```
sh load_amazon-review/load.sh
```

Then, in the NGCF folder, run

```bash
mkdir profile
mkdir mprof
```

```bash
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --batch_size 1024 --epoch 400 --verbose 1 --mess_dropout [0.1,0.1,0.1] --model_type 2 --profile 1
```
In the bash scirpt, ```profile = 1``` stands for profiling only in forward part.
```model_type``` could be chosen from 0-5. 0 is the original implementation of dgl.


Run the following code to get experimental result automatically.
```bash
python run.py
```

Memory profiling results are shown in the mprof directory. Results of time occupancy regard to different operations are shown in the profile directory.
