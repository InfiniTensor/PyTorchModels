#!/bin/bash

set -e

export ASCEND_RT_VISIBLE_DEVICES=1

saved_dir=$1
num_episodes=$2
lr=$3

python main.py \
       --save_path $saved_dir \
       --num_episodes $num_episodes \
       --lr=$lr

# bash run_train.sh checkpoints 100 0.0001
