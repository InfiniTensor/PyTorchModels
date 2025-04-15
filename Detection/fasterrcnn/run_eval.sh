#!/bin/bash

set -e

export ASCEND_RT_VISIBLE_DEVICES=1

# 使用环境变量，如果没有提供则使用默认路径
data_dir=${DATA_DIR:-""}
ckpt_dir=${CKPT_DIR:-"./"}

# 确保数据集路径存在
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist."
    exit 1
fi

# 检查数据目录是否存在
if [ -e "$data_dir" ]; then
    echo "$data_dir exists"
fi

ckpt_path="$ckpt_dir/fasterrcnn.pth"

ckpt_url="https://cloud.tsinghua.edu.cn/seafhttp/files/1db6485b-ef12-42b3-b9e7-a9b86be648bc/fasterrcnn.pth"

# 检查模型文件是否存在
if [ -e "$ckpt_path" ]; then
    echo "$ckpt_path exists"
else
    echo "Download $ckpt_path from url $ckpt_url"
    wget "$ckpt_url" -O "$ckpt_path"
fi

echo "Evaluate FasterRCNN START"
# python eval.py main --load-path="$ckpt_path" --voc_data_dir="$data_dir/VOC2007" --test_num_workers=2
python eval.py main --voc_data_dir="$data_dir/VOC2007" --test_num_workers=2
echo "Evaluate FasterRCNN FINISHED"

# rm $ckpt_path

# mAP=0.6975
