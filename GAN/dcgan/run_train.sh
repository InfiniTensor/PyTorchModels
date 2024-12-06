#!/bin/bash
set -e

if [ -e "../data/lsun" ]; then
	echo "../data/lsun exists"
else
	ln -s /data1/shared/Dataset/lsun ../data
fi

if [ -e "./output" ]; then
	rm -rf "./output"
else
	mkdir output
fi

export CUDA_VISIBLE_DEVICES=0

python3 train.py --dataset lsun --dataroot ../data/lsun/ --cuda
