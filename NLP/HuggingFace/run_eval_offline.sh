# 无互联网连接时
export CUDA_VISIBLE_DEVICES=0
WORKING_DIR=$(pwd)

torchrun \
    --nproc_per_node=1 \
    qa.py \
    --model_name_or_path $WORKING_DIR/bert-base-uncased \
    --config_name $WORKING_DIR/bert-base-uncased \
    --tokenizer_name $WORKING_DIR/bert-base-uncased/ \
    --train_file $WORKING_DIR/squad/train-v2.0.json \
    --validation_file $WORKING_DIR/squad/dev-v2.0.json \
    --test_file $WORKING_DIR/squad/dev-v2.0.json \
    --version_2_with_negative \
    --per_device_eval_batch_size 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --do_eval \
    --output_dir /tmp/debug_squad/ \

