#!/bin/bash

cd "$(dirname "$0")"

data_dir="../data/sas-55m-20k"

if [ ! -f "$data_dir/scales.json" ]; then
    # If scales.json does not exist, move it from sasformer/data to data_dir
    cp ../data/scales.json "$data_dir"
fi

mkdir -p results

python ../sasformer/results.py --ckpt_path ../checkpoints/final.ckpt --data_dir ../data/sas-55m-20k/ --batch_size 1500 --accelerator gpu