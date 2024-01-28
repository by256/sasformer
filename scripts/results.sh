#!/bin/bash

cd "$(dirname "$0")"

# CHANGE THIS TO THE PATH WHERE YOU SAVED THE SAS-55M-20K DATASET
data_dir="../data/sas-55m-20k"

if [ ! -f "$data_dir/scales.json" ]; then
    # If scales.json does not exist, move it from sasformer/data to data_dir
    cp ../data/scales.json "$data_dir"
fi

mkdir -p ../results
mkdir -p ../checkpoints

if [ ! -f "../checkpoints/final.ckpt" ]; then
    # If final.ckpt does not exist, download it using wget
    echo "Downloading Model Checkpoint..."
    wget -O ../checkpoints/final.ckpt "https://huggingface.co/by256/sasformer/resolve/main/final.ckpt?download=true"
fi

python ../sasformer/results.py --ckpt_path ../checkpoints/final.ckpt --data_dir ../data/sas-55m-20k/ --batch_size 1500 --accelerator gpu