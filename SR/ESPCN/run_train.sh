#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if [ -d "data/train/SRF_2/data" ] && [ "$(ls data/train/SRF_2/data/ | wc -l)" -gt 0 ]; then
    echo "Train data exists, skip data generation"
else
    python data_utils.py --upscale_factor 2
fi

python train.py --upscale_factor 2 --num_epochs 2
