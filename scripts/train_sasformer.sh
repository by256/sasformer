#!/bin/bash

cd "$(dirname "$0")"

data_dir="../data/sas-55m-20k"

python ../sasformer/train.py --data_dir "$data_dir" --batch_size 96 --num_latents 48 --latent_dim 1024 --enc_num_self_attn_per_block 12 --enc_num_self_attn_heads 4 --enc_num_cross_attn_heads 4 --enc_dropout 0.05 --model_dec_num_heads 4 --model_dec_widening_factor 3 --model_dec_dropout 0.45 --param_dec_num_heads 2 --param_dec_widening_factor 1 --param_dec_dropout 0.05 --max_epochs 200 --lr 5e-3 --accumulate_grad_batches 19 --subsample 4096 