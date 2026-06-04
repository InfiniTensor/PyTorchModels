# 有互联网连接时
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# 检查软连接是否已经存在了
if [ -e "../data/squad" ]; then
    echo "../data/squad exists"
else
    ln -s /data-aisoft/Dataset/squad ../data/squad
fi

export SQUAD_PATH="../data/squad"

PYTHONUNBUFFERED=1 python3 qa.py \
    --model_name_or_path bert-base-uncased \
    --train_file $SQUAD_PATH/train-v2.0.json \
    --validation_file $SQUAD_PATH/dev-v2.0.json \
    --version_2_with_negative \
    --per_device_train_batch_size 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --max_train_samples 1000 \
    --do_train \
    --output_dir /tmp/debug_squad/
