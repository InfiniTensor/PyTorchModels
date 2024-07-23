#!/bin/bash

ckpt=$2
dataset=$1

export MLU_VISIBLE_DEVICES=2


python -W ignore infer.py \
       --dataset $dataset \
       --model_path $ckpt \
       --infer \
       # --infer_data $3

# /dataset/LSTM_data/complete_data.csv md.pth "2021-08-31 13:15:00",250994.7469
