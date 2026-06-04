# 有互联网连接时
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
unset HF_ENDPOINT

PYTHONUNBUFFERED=1 python3 qa.py \
    --model_name_or_path /data-aisoft/InfiniTrain/bolunz/Models/bert-base-uncased \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --max_train_samples 1000 \
    --do_train \
    --output_dir /tmp/debug_squad/
