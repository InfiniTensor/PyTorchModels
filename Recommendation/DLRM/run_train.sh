#!/bin/bash

# 脚本用于分布式训练模型，并在训练过程中输出时间和性能结果
# 必须传递的数据集路径：PYTORCH_TRAIN_DATASET
# 其他参数可以通过修改脚本中的变量配置

set -e  # 一旦出现错误，退出脚本

# 加速卡设备配置
export ASCEND_RT_VISIBLE_DEVICES=1,2

# 获取当前脚本所在目录
CUR_DIR=$(cd $(dirname $0); pwd)

# 默认配置
DATASET_DIR=${DATA_DIR:-""}  # /data1/shared/Dataset/ml-20mx4x16
THRESHOLD=${THRESHOLD:-1.0}  # 默认阈值
ckp_dir=${CUR_DIR}/checkpoints  # 检查点保存路径
cache_dir=${CUR_DIR}/data  # 缓存目录
nproc_per_node=2  # 每个节点的进程数
device='gpu'  # 使用GPU训练

# 检查数据集路径是否存在
if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: Directory ${DATASET_DIR} does not exist."
    exit 1
fi

# 开始计时
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# 执行分布式训练
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
    --device ${device} \
    --do_train \
    --save_ckp 1 \
    --ckpdir ${ckp_dir} \
    --cachedir ${cache_dir} \
    --multiprocessing-distributed \
    --iters -1 \
    --use_amp 1

# 结束计时
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# 计算并报告结果
result=$((end - start))
result_name="recommendation"

echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"

