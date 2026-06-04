#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else 
    ln -s /data-aisoft/Dataset/VOCdevkit ../data/VOCdevkit
fi

python $SCRIPT_DIR/deeplab.py \
        --train-batch-size 4 \
        --train-epochs 10 \
        --mode train \
        --image-size 256 \


