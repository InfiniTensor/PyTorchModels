#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else 
    ln -sfn /data-aisoft/Dataset/VOCdevkit ../data/VOCdevkit
fi

python $SCRIPT_DIR/fcn.py \
        --infer-batch-size 1 \
        --image-size 256 \
        --mode infer \
        --max_batches 10 \
