#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  /data1/shared/Dataset/VOCdevkit 

# 确保脚本在遇到错误时停止执行
set -e

export CUDA_VISIBLE_DEVICES=0

# 检查软连接是否已经存在了
if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else
    # 创建软连接
    ln -s  /data1/shared/Dataset/VOCdevkit ../data
fi

# execute create_data_lists.py
echo "executing create_data_lists.py..."
python create_data_lists.py

# execute train.py
echo "executing train.py..."
python train.py

# execute detect.py
echo "executing detect.py..."
python eval.py

echo "finished"
