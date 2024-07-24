#!/bin/bash

if [ -e "../data/data_thchs30" ]; then
    echo "../data/data_thchs30 exists"
else
    # 创建软连接
    # ln -s /data1/shared/Dataset/data_thchs30 ../data
    mkdir ../data
    ln -s /home/qinyiqun/data_thchs30 ../data/data_thchs30
fi

python data_preprocess.py

python train.py