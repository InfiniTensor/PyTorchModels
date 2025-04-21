#!/bin/bash

set -e 

export ASCEND_RT_VISIBLE_DEVICES=2

dataset=$1
epoch=$2
batch_size=$3
learning_rate=$4

python -W ignore train.py \
       --dataset $dataset \
       --epoch $epoch \
       --batch_size $batch_size \
       --lr $learning_rate \

# bash run_train.sh ../data/complete_data.csv 200 512 0.0001
