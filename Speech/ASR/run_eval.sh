#!/bin/bash

if [ -e "./cache/data_thchs30" ]; then
    echo "./cache/data_thchs30 exists"
else
    # 创建软连接
    mkdir ./cache
    ln -s /data1/shared/Dataset/data_thchs30 ./cache/data_thchs30
    # ln -s /home/qinyiqun/data_thchs30 ../data/data_thchs30
fi

python data_preprocess.py

python eval.py