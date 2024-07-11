#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/coco exists"
else
    ln -s /data1/shared/Dataset/VOCdevkit ../data
fi

echo "Training FasterRCNN START"
python train.py train --voc_data_dir=../data/VOCdevkit/VOC2007
echo "Training FasterRCNN FINISHED"