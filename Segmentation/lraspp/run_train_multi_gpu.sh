#!/bin/bash

# 多卡训练脚本
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1}

if [ -e "../data/data_thchs30" ]; then
    echo "../data/data_thchs30 exists"
else
    ln -sf /data-aisoft/Dataset/data_thchs30 ../data_thchs30
fi

if [ -e "./cache" ]; then
    echo "./cache exists"
else
    mkdir cache
fi

python data_preprocess.py

torchrun --nproc_per_node=4 train.py
