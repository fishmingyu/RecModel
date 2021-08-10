import os
dataset = ['gowalla', 'amazon-book']
hidden_size = [(2 ** i) for i in range(5, 8)]
model_type = range(6)
for i in model_type:
    for dat in dataset:
        os.system("python main.py " + "--dataset " + dat + " --model_type " + str(i) + " --profile 1")
