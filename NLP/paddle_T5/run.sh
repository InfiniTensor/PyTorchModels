#!/bin/bash

# 进入paddlenlp目录
cd paddlenlp
# 本地安装
pip install -r requirements.txt
pip install -e .
# 返回初始目录
cd ..

# 确保处在GLUE文件夹
cd TASK/GLUE
# 安装依赖
pip install -r requirements.txt
# 运行训练
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name mnli \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir outputs/mnli/ \
    --device gpu \
    --num_workers 2
