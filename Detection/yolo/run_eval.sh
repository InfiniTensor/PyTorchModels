
#!/bin/bash

# 确保脚本在遇到错误时停止执行
#set -e

# # 配置环境
# echo "配置环境..."
# #pip install -r requirements.txt

set -e


export CUDA_VISIBLE_DEVICES=0

if [ -e "../data/coco" ]; then
    echo "../data/coco exists"
else
    ln -s /data1/shared/Dataset/coco/ ../data
fi

# CP Arial.ttf to local
mkdir -p ~/.config/Ultralytics/
cp ./Arial.ttf ~/.config/Ultralytics/Arial.ttf

declare -A MODELS

MODELS["yolov5n"]="https://cloud.tsinghua.edu.cn/seafhttp/files/ed37cef3-e656-4bbe-868d-b640ca9645f1/yolov5n.pt"
MODELS["yolov5s"]="https://cloud.tsinghua.edu.cn/seafhttp/files/6739c917-1494-4c26-93bd-4f7d50f02f2e/yolov5s.pt"
MODELS["yolov5m"]="https://cloud.tsinghua.edu.cn/seafhttp/files/3f3a1cf9-3e74-43bd-9ed1-dfd9396246c6/yolov5m.pt"
MODELS["yolov5l"]="https://cloud.tsinghua.edu.cn/seafhttp/files/91ba88be-99b8-47fd-8c34-8914f66d840e/yolov5l.pt"
MODELS["yolov5x"]="https://cloud.tsinghua.edu.cn/seafhttp/files/927abf57-76c8-4f27-97a7-948e9d2f7f90/yolov5x.pt"

# 读取用户输入 
model_option=$1

# 检查输入是否有效  
is_valid=false  
for model in "${!MODELS[@]}"; do  
    if [[ "$model_option" == "$model" ]]; then  
        is_valid=true  
        break  
    fi  
done  

# 检查输入是否有效，并据此运行脚本  
if $is_valid; then  

    model_path="./${model_option}.pt"

    if [ -e $model_path ]; then
        echo "$model_path exists"
    else
        echo "Download $model_option from url ${MODELS[$model_option]}"
        # Download model ckpt from server
        wget "${MODELS[$model_option]}"
    fi

    echo "Evaluate $model_option START" 


    if [ -f $model_path ]; then  
        # 推理  
        python3 val.py --weights $model_path --data coco.yaml --img 640 
    else  
        echo "Model path $model_path not exists "  
    fi  

    # rm $model_path
    echo "Evaluate $model_option FINISHED"
else  
    echo "Choose model in yolov5n yolov5s yolov5m yolov5x yolov5n"  

fi

# mAP50: 0.565 mAP50-95: 0.371
