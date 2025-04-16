# 无互联网连接时
export ASCEND_RT_VISIBLE_DEVICES=1,2,3
# export HF_ENDPOINT=https://hf-mirror.com

# 检查软连接是否已经存在了
if [ -e "../data/squad" ]; then
    echo "../data/squad exists"
else
    # 创建软连接
    ln -s /data1/shared/Dataset/squad ../data/squad
fi

export SQUAD_PATH="../data/squad"

# 模型文件目录路径，路径下需要包含：
#   config.json
#   pytorch_model.bin
#   tokenizer_config.json
#   tokenizer.json
#   vocab.txt

model=$1

declare -A MODELS

MODELS["bert-base-uncased"]="https://cloud.tsinghua.edu.cn/seafhttp/files/fb07c1c6-97c5-4b6a-ab5e-c969487a2b0e/bert-base-uncased.zip"
MODELS["albert-base-v2"]="https://cloud.tsinghua.edu.cn/seafhttp/files/dea2b22e-2cd8-42a9-8994-7894c8a5fa72/albert-base-v2.zip"

# MODEL_PATH="./bert-base-uncased"
MODEL_PATH="./${model}"
MODEL_URL=${MODELS[$model]}

if [ -e ${MODEL_PATH} ]; then
    echo "${MODEL_PATH} exists"
else 
    echo "Download ${MODEL_PATH}.zip from url $MODEL_URL"
    wget $MODEL_URL
    unzip ${MODEL_PATH}.zip && rm ${MODEL_PATH}.zip
fi

torchrun \
    --nproc_per_node=1 \
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
    --output_dir ./tmp/debug_squad/ \

if [ -e ./tmp ]; then
    rm -rf ./tmp
fi
