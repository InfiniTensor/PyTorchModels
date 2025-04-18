#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=1

python data_utils.py --upscale_factor 2

python train.py --upscale_factor 2 --num_epochs 2
