#!/bin/bash

# 确保脚本在遇到错误时停止执行
#set -e

# # 配置环境
# echo "配置环境..."
# #pip install -r requirements.txt

set -e

if [ -e "./runs" ]; then
    rm -rf "./runs"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ -e "../data/coco" ]; then
    echo "../data/coco exists"
else
    ln -s /data1/shared/Dataset/coco/ ../data
fi

# CP Arial.ttf to local
mkdir -p ~/.config/Ultralytics/
cp ./Arial.ttf ~/.config/Ultralytics/Arial.ttf

MODELS=("yolov5n"
        "yolov5s"
        "yolov5m"
        "yolov5l"
        "yolov5x")

# 读取用户输入 
model_option=$1

# 检查输入是否有效  
is_valid=false  
for model in "${MODELS[@]}"; do  
    if [[ "$model_option" == "$model" ]]; then  
        is_valid=true  
        break  
    fi  
done  

# 检查输入是否有效，并据此运行脚本  
if $is_valid; then  

    # 训练
    echo "Training $model_option START" 
#    python3 train.py --data coco.yaml --epochs 300 --weights '' --cfg "${model_option}.yaml"  --batch-size 64
    python -m torch.distributed.run \
        --nproc_per_node 4 train.py \
        --batch 64 \
        --img 640 \
        --epoch 25 \
        --data coco.yaml \
        --weights "" \
        --cfg "models/${model_option}.yaml" \
        --device 0,1,2,3 \
        --nosave \
        --noval \
        --workers 16 
    echo "Training $model_option FINISHED"

else  
    echo "Choose model in yolov5n yolov5s yolov5m yolov5x yolov5n"  
fi

