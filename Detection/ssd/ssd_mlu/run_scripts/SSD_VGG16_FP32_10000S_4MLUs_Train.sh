#!/bin/bash
# 这个脚本用于运行 Python 脚本
# 数据集路径通过环境变量传入

set -e

CUR_DIR=$(cd $(dirname $0);pwd)
export MLU_VISIBLE_DEVICES=1,2,3,4
export PYTHONPATH=$CUR_DIR/../:$PYTHONPATH
ITERATIONS=$1

SAVE_PATH="/workspace/PyTorchModels/Detection/ssd/output/train/MLU_SSD_VGG16_${ITERATIONS}S_FP32_4MLUs"
MODEL_DIR="/workspace/PyTorchModels/Detection/ssd/mlu_ssd/models"
BASENET="/workspace/PyTorchModels/Detection/ssd/vgg16_reducedfc.pth"

mkdir -p $SAVE_PATH

python -W ignore $CUR_DIR/../models/SSD_VGG16_train.py \
    --dataset_root /dataset/VOC2007 \
    --save_folder $SAVE_PATH \
    --dataset VOC \
    --seed 42 \
    --iters $ITERATIONS \
    --device mlu \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --batch_size $2 \
    --lr $3 \
    --dist-backend cncl \
    --basenet $BASENET \
    #--dist-url "auto"

python $MODEL_DIR/SSD_VGG16_test.py \
    --trained_model $SAVE_PATH/mlu_weights_ssd300_VOC_$ITERATIONS.pth \
    --voc_root /dataset/VOC2007 \
    --device mlu
