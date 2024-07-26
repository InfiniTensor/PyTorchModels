#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1

CUR_DIR=$(cd $(dirname $0);pwd)

PYTORCH_TRAIN_DATASET=/data1/shared/Dataset

THRESHOLD=1.0
ckp_dir=${CUR_DIR}/checkpoints
cache_dir=${CUR_DIR}/data
nproc_per_node=2
device='gpu'

DATASET_DIR=${PYTORCH_TRAIN_DATASET}/ml-20mx4x16

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

	python -m torch.distributed.launch \
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
        --device $device \
        --do_train \
        --save_ckp 1 \
        --ckpdir ${ckp_dir} \
        --cachedir ${cache_dir} \
        --multiprocessing-distributed \
        --iters -1 \
        --use_amp 1

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"

	# report result
	result=$(( $end - $start ))
	result_name="recommendation"

	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi
