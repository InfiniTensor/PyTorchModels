#!/bin/bash

set -e

export MLU_VISIBLE_DEVICES=0

saved_dir=$1
num_episodes=$2
lr=$3

python main.py \
       --save_path $saved_dir \
       --num_episodes $num_episodes \
       --lr=$lr

# bash run_train.sh checkpoints 100 0.0001
