#!/bin/bash

# 确保脚本在遇到错误时停止执行
#set -e

# # 配置环境
# echo "配置环境..."
# #pip install -r requirements.txt

# 数据集准备
echo "数据集准备..."
if [ -d "datasets" ]; then
    rm -rf datasets
fi
mkdir datasets
cd datasets
ln -s /data1/shared/Dataset/coco/ ../data
cd ../

mkdir -p ~/.config/Ultralytics/
cp ./Arial.ttf ~/.config/Ultralytics/Arial.ttf

VALID_OPTIONS=("yolov5s" "yolov5m" "yolov5l" "yolov5x" "yolov5n") 
# 读取用户输入 
read -p "请输入一个模型选项（yolov5s, yolov5m, yolov5l, yolov5x, yolov5n）: " model_option 
# 检查输入是否有效  
is_valid=false  
for option in "${VALID_OPTIONS[@]}"; do  
    if [[ "$model_option" == "$option" ]]; then  
        is_valid=true  
        break  
    fi  
done  

# 检查输入是否有效，并据此运行脚本  
if $is_valid; then  
    echo "使用模型选项: $model_option"  
    # 推理
    echo "推理..."
    # 检查 detect.py 是否存在以及 weights 文件是否存在  
    if [ -f "${model_option}.pt" ]; then  
        # 推理  
        python3 detect.py --weights "${model_option}.pt" --source data/images/zidane.jpg  
    else  
        echo "模型权重文件不存在。"  
    fi  
    # 训练
    echo "训练..." 
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
        --workers 16 \
        --profile
    # 验证
    echo "验证..."
    if [ -f "${model_option}.pt" ]; then  
        # 推理  
        python3 val.py --weights "${model_option}.pt" --data coco.yaml --img 640 
    else  
        echo "模型权重文件不存在。"  
    fi  
    
else  
    echo "输入的模型选项无效，请确保输入yolov5s, yolov5m, 或 yolov5l中的一个。"  
fi

echo "所有脚本执行完毕。"
