#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  /VOCdevkit/

# 确保脚本在遇到错误时停止执行
set -e

# 运行train.py
echo "正在运行 train.py..."
python train.py --dataset_path ./data --VOC_year 2007

# 运行train.py
echo "正在运行 eval.py..."
python eval.py --dataset_path ./data --VOC_year 2007

echo "所有脚本执行完毕。"