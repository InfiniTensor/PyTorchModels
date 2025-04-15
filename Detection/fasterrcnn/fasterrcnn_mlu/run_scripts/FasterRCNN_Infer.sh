#!/bin/bash
set -e
export MLU_VISIBLE_DEVICES=1
pushd ../models
# model_final.pth基于FasterRCNN_FP32_20000S_8MLUs_Train.sh训练得到，路径见脚本内部指定
python FasterRCNN_infer.py  --config-file configs/FasterRCNN_Eval.yaml \
                            --ckpt $1 \
                            MODEL.DEVICE mlu \
                            DATASETS.TEST "('coco_2017_val',)" \
                            TEST.IMS_PER_BATCH $3
#../final.pth /dataset/COCO2017 32
