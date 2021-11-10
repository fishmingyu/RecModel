#!/bin/bash
python model.py ./datasets_pickle/ml-1m.pkl --num-epochs 5 --num-workers 1 --device cpu --hidden-dims 64 --batches-per-epoch 1 --batch-size 3706