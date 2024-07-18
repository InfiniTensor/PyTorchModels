#!/bin/bash

# 脚本用于运行 Torchvision 中所有分类模型的的训练
# 数据集放在 /data1/shared/Dataset/imagenet2012


# 检查软连接是否已经存在了
if [ -e "../data/imagenet2012" ]; then
    echo "../data/imagenet2012 exists"
else
    # 创建软连接
    ln -s /data1/shared/Dataset/imagenet2012 ../data
fi


# 模型列表
models=(
    alexnet
    convnext_base
    convnext_large
    convnext_small
    convnext_tiny
    densenet121
    densenet161
    densenet169
    densenet201
    efficientnet_b0
    efficientnet_b1
    efficientnet_b2
    efficientnet_b3
    efficientnet_b4
    efficientnet_b5
    efficientnet_b6
    efficientnet_b7
    googlenet
    inception_v3
    mnasnet0_5
    mnasnet0_75
    mnasnet1_0
    mnasnet1_3
    mobilenet_v2
    mobilenet_v3_large
    mobilenet_v3_small
    regnet_x_16gf
    regnet_x_1_6gf
    regnet_x_32gf
    regnet_x_3_2gf
    regnet_x_400mf
    regnet_x_800mf
    regnet_x_8gf
    regnet_y_128gf
    regnet_y_16gf
    regnet_y_1_6gf
    regnet_y_32gf
    regnet_y_3_2gf
    regnet_y_400mf
    regnet_y_800mf
    regnet_y_8gf
    resnet101
    resnet152
    resnet18
    resnet34
    resnet50
    resnext101_32x8d
    resnext50_32x4d
    shufflenet_v2_x0_5
    shufflenet_v2_x1_0
    shufflenet_v2_x1_5
    shufflenet_v2_x2_0
    squeezenet1_0
    squeezenet1_1
    vgg11
    vgg11_bn
    vgg13
    vgg13_bn
    vgg16
    vgg16_bn
    vgg19
    vgg19_bn
    vit_b_16
    vit_b_32
    vit_l_16
    vit_l_32
    wide_resnet101_2
    wide_resnet50_2
)

# 环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 遍历所有模型
for model in "${models[@]}"; do
    echo "Running model: $model START"
    python main.py \
        -a $model \
        --dist-backend 'nccl' \
        --dist-url "tcp://localhost:8828" \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --batch-size 64 \
        ../data/imagenet2012 &

    # 获取进程ID并保存
    pid=$!

    sleep 600 

    # 杀死进程及其子进程
    pkill -P $pid
    kill $pid

    echo "Running model: $model FINISHED"
    
    # 等待输出缓冲区冲刷
    sleep 5

done



