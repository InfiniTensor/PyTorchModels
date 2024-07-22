#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

set -e

export CUDA_VISIBLE_DEVICES=0

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else 
    ln -s /data1/shared/Dataset/VOCdevkit ../data/VOCdevkit
fi

python $SCRIPT_DIR/deeplab.py \
        --infer-batch-size 1 \
        --image-size 256 \
        --mode infer \