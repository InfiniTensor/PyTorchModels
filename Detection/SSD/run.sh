#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  /VOCdevkit/

# 确保脚本在遇到错误时停止执行
set -e

# execute create_data_lists.py
echo "executing create_data_lists.py..."
python create_data_lists.py

# execute train.py
echo "executing train.py..."
python train.py

# execute detect.py
echo "executing detect.py..."
python detect.py

echo "finished"
