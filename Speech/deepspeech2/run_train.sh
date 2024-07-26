#!/bin/bash

if [ -e "../data/data_thchs30" ]; then
    echo "../data/data_thchs30 exists"
else
    ln -s /data1/shared/Dataset/data_thchs30 ../data/data_thchs30
fi

if [ -e "./cache" ]; then
    echo "./cache exists"
else
    mkdir cache
fi

python data_preprocess.py

python train.py
