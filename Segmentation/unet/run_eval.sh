
#!/bin/bash

# 这个脚本用于运行Python脚本
# 数据集放在同级目录下  ./data

# 确保脚本在遇到错误时停止执行
set -e

export CUDA_VISIBLE_DEVICES=0

if [ -e "../data/VOCdevkit" ]; then
    echo "../data/VOCdevkit exists"
else 
    ln -s /data1/shared/Dataset/VOCdevkit ../data/VOCdevkit
fi

# 运行train.py
echo "Evaluating UNet START"

# 参数解释
# dataset_path 数据集保存位置
# dataset_path
#   - VOCtrainval_06-Nov-2007.tar
# VOC_year 指定解析数据集的方法
# input_size 输入模型的图像大小
# classes 模型输出类别数量
python eval.py \
       --device cuda \
       --input_size 256 \
       --classes 22 \
       --dataset_path ../data \
       --VOC_year 2007 \

echo "Evaluating UNet FINISHED"