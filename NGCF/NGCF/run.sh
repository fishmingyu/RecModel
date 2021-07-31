mkdir profile

python main.py --dataset gowalla --regs [1e-5] --embed_size 32 --layer_size [32,32,32] --lr 0.0001 --save_flag 1 --batch_size 1024 --epoch 400 --verbose 1 --mess_dropout [0.1,0.1,0.1] --model_type 0 --profile 0
python main.py --dataset gowalla --regs [1e-5] --embed_size 32 --layer_size [32,32,32] --lr 0.0001 --save_flag 1 --batch_size 1024 --epoch 400 --verbose 1 --mess_dropout [0.1,0.1,0.1] --model_type 1 --profile 0
python main.py --dataset gowalla --regs [1e-5] --embed_size 32 --layer_size [32,32,32] --lr 0.0001 --save_flag 1 --batch_size 1024 --epoch 400 --verbose 1 --mess_dropout [0.1,0.1,0.1] --model_type 2 --profile 0
python main.py --dataset gowalla --regs [1e-5] --embed_size 32 --layer_size [32,32,32] --lr 0.0001 --save_flag 1 --batch_size 1024 --epoch 400 --verbose 1 --mess_dropout [0.1,0.1,0.1] --model_type 3 --profile 0