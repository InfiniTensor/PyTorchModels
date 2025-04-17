#!/bin/bash

set -e

export ASCEND_RT_VISIBLE_DEVICES=1

model_path=$1

if [ -e "$model_path" ]; then
    echo "$model_path exists, use it"
else
    echo "$model_path not exists, please train first"
fi

python -W ignore main.py \
       --model_path $model_path \
       --infer 

# bash run_eval.sh ./checkpoints/20.pth
