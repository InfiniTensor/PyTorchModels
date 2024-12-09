#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python data_utils.py --upscale_factor 2

python train.py --upscale_factor 2 --num_epochs 2
