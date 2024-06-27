# 有互联网连接时
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

torchrun \
    --nproc_per_node=4 \
    qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --do_train \
    --output_dir /tmp/debug_squad/ 
