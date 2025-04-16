#!/bin/bash

set -e

export ASCEND_RT_VISIBLE_DEVICES=1

# 使用环境变量，如果没有提供则使用默认路径
data_dir=${DATA_DIR:-""} # data1/shared/Dataset/VOCdevkit

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# 检查数据目录是否存在
if [ -e "$data_dir" ]; then
    echo "$data_dir exists"
fi

echo "Training FasterRCNN START"
python train.py train --voc_data_dir="$data_dir/VOC2007" --num_workers=2 --test_num_workers=2
echo "Training FasterRCNN FINISHED"
