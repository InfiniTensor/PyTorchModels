#!/bin/bash

# 多卡训练脚本
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

torchrun --nproc_per_node=4 pmnist_test.py
