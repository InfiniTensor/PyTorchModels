#!/bin/bash

set -e

export ASCEND_RT_VISIBLE_DEVICES=2

ckpt=$2
dataset=$1

if [ -e "$ckpt" ]; then
       echo "$ckpt exists, use it"
else
       echo "$ckpt not exists, please run train first"
fi

python -W ignore eval.py \
       --dataset $dataset \
       --model_path $ckpt \

# bash run_eval.sh ../data/complete_data.csv ./checkpoints/lstm_best.pt
