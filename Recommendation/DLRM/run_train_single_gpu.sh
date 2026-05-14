#!/bin/bash

# 单卡训练版
set -e

export CUDA_VISIBLE_DEVICES=0

CUR_DIR=$(cd $(dirname $0); pwd)

DATASET_DIR=${DATA_DIR:-""}
THRESHOLD=${THRESHOLD:-1.0}
ckp_dir=${CUR_DIR}/checkpoints
cache_dir=${CUR_DIR}/data
nproc_per_node=1
device='gpu'

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: Directory ${DATASET_DIR} does not exist."
    exit 1
fi

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

PYTHONUNBUFFERED=1 python -m torch.distributed.launch \
    --nproc_per_node=${nproc_per_node} \
    --master_port 29501 \
    --use_env \
    ncf.py \
    --data ${DATASET_DIR} \
    -l 0.0002 \
    -b 65536 \
    --layers 256 256 128 64 \
    -f 64 \
    --seed 0 \
    --threshold ${THRESHOLD} \
    --user_scaling 4 \
    --item_scaling 16 \
    --cpu_dataloader \
    --workers 8 \
    --random_negatives \
    --device ${device} \
    --do_train \
    --save_ckp 1 \
    --ckpdir ${ckp_dir} \
    --cachedir ${cache_dir} \
    --multiprocessing-distributed \
    --iters -1 \
    --use_amp 1

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

result=$((end - start))
result_name="recommendation"

echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
