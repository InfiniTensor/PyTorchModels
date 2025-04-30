#!/bin/bash

#基于pytorch训练模型
#get data
bash getdata.sh

#train data
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_enwik8_base.sh train --work_dir PATH_TO_WORK_DIR

#evaluate model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_enwik8_base.sh eval --work_dir PATH_TO_WORK_DIR

