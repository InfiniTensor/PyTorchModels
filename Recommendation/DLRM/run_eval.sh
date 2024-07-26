#!/bin/bash
set -e

PYTORCH_INFER_CHECKPOINT=$1
PYTORCH_TRAIN_DATASET=/data1/shared/Dataset

CUR_DIR=$(cd $(dirname $0);pwd)
cache_dir=${CUR_DIR}/data
DATASET_DIR=${PYTORCH_TRAIN_DATASET}/ml-20mx4x16

if [ -d ${DATASET_DIR} ]
then
	python ncf.py \
		--data ${DATASET_DIR} \
		--resume ${PYTORCH_INFER_CHECKPOINT} \
		-l 0.0002      \
		-b 65536      \
		--layers 256 256 128 64  \
		-f 64     \
		--seed 0  \
		--save_ckp 1 \
		--threshold 1.0 \
		--user_scaling 4  \
		--item_scaling 16 \
		--cpu_dataloader   \
		--random_negatives  \
		--device gpu \
		--workers 8 \
		--do_predict \
		--cachedir ${cache_dir} \
		--multiprocessing-distributed
else
	echo "Directory ${DATASET_DIR} does not exist"
fi
