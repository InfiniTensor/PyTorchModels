#!/bin/bash

if [ -e "../data/data_thchs30" ]; then
    echo "../data/data_thchs30 exists"
else
    ln -s /data-aisoft/Dataset/data_thchs30 ../data/data_thchs30
fi

if [ -e "./cache/manifest.test" ]; then
    echo "./cache exists, skip data_preprocess"
else
    mkdir -p cache
    python data_preprocess.py
fi

python eval.py
