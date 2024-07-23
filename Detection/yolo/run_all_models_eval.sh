#!/bin/bash

if [ -e "./runs" ]; then
    rm -rf "./runs"
fi

export CUDA_VISIBLE_DEVICES=0

if [ -e "../data/coco" ]; then
    echo "../data/coco exists"
else
    ln -s /data1/shared/Dataset/coco/ ../data
fi

# CP Arial.ttf to local
mkdir -p ~/.config/Ultralytics/
cp ./Arial.ttf ~/.config/Ultralytics/Arial.ttf

models=(
    "yolov5n"
    "yolov5s"
    "yolov5m"
    "yolov5l"
    "yolov5x"
)

echo "Evaluating start: $(date +'%m/%d/%Y %T')"

for model in "${models[@]}"; do

    model_path="./${model}.pt"

    if [ -e $model_path ]; then
        echo "$model_path exists"
    else # TODO: saving ckpt to somewhere
        echo "Please download ckpt"
    fi

    echo "Evaluating $model start: $(date +'%m/%d/%Y %T')"

    if [ -f $model_path ]; then  
        # 推理  
        python3 val.py --weights $model_path --data coco.yaml --img 640 
    else  
        echo "Model path $model_path not exists "  
    fi  

    echo "Evaluating $model finish: $(date +'%m/%d/%Y %T')"

    sleep 5

done
