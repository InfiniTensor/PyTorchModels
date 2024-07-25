#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

if [ -e "../data/data_thchs30" ]; then
    echo "../data/data_thchs30 exists"
else
    # 创建软连接
    ln -s /data/Dataset/data_thchs30 ./cache/data_thchs30
fi

if [ -e "./cache" ]; then
    echo "./cache exists"
else
    mkdir cache
fi

python data_preprocess.py

python -u train.py
