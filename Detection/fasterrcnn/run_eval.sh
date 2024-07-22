#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/coco exists"
else
    ln -s /data1/shared/Dataset/VOCdevkit ../data
fi

ckpt_path="./fasterrcnn.pth"

ckpt_url="https://cloud.tsinghua.edu.cn/seafhttp/files/1db6485b-ef12-42b3-b9e7-a9b86be648bc/fasterrcnn.pth"

if [ -e $ckpt_path ]; then
    echo "$ckpt_path exists"
else
    echo "Download $ckpt_path from url $ckpt_url"
    wget $ckpt_url
fi

echo "Evaluate FasterRCNN START"
python eval.py main --load-path=$ckpt_path
echo "Evaludate FasterRCNN FINISHED"

# rm $ckpt_path

# mAP=0.6975
