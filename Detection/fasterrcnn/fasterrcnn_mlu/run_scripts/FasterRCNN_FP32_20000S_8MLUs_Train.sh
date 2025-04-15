#!/bin/bash
set -e
export MLU_VISIBLE_DEVICES=0,1,2,3
DEVICES_NUMS=$(echo $MLU_VISIBLE_DEVICES | awk -F "," '{print NF}')
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
pushd ../models
python -m torch.distributed.launch  --master_addr=$MASTER_ADDR \
                                    --nproc_per_node=$DEVICES_NUMS \
                                    --master_port=$PORT \
                                    FasterRCNN_train.py \
                                    --config-file configs/FasterRCNN_FP32_Train.yaml \
                                    --prefix faster \
                                    DATASETS.TRAIN "('coco_2017_train',)" \
				    DATASETS.TEST "('coco_2017_val',)" \
                                    MODEL.DEVICE mlu \
				    MODEL.WEIGHT /workspace/PyTorchModels/Detection/fasterrcnn/fasterrcnn_mlu/run_scripts/R-101.pkl \
                                    SOLVER.MAX_ITER $1 \
                                    SOLVER.IMS_PER_BATCH $2 \
                                    SOLVER.BASE_LR $3
# 10000 32 0.00002
