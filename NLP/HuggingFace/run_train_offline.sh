# 无互联网连接时
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SQUAD_PATH=./squad

# 模型文件目录路径，路径下需要包含：
#   config.json
#   pytorch_model.bin
#   tokenizer_config.json
#   tokenizer.json
#   vocab.txt
MODEL_PATH=./bert-base-uncased

torchrun \
    --nproc_per_node=4 \
    qa.py \
    --model_name_or_path $MODEL_PATH \
    --config_name $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --train_file $SQUAD_PATH/train-v2.0.json \
    --validation_file $SQUAD_PATH/dev-v2.0.json \
    --test_file $SQUAD_PATH/dev-v2.0.json \
    --version_2_with_negative \
    --per_device_train_batch_size 10 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --do_train \
    --output_dir /tmp/debug_squad/ \

