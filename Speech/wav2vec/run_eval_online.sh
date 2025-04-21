#!/usr/bin/env bash

if [ -e "../data/LibriSpeech" ]; then
    echo "../data/LibriSpeech exists"
else
    ln -s /data1/shared/Dataset/librispeech/LibriSpeech ../data/LibriSpeech
fi

export ASCEND_RT_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com
export LIBRISPEECH_PATH="../data/LibriSpeech"

CACHE_PATH="./cache"
mkdir -p $CACHE_PATH

python speech_recognition.py \
	--dataset_name="librispeech_asr" \
	--model_name_or_path="facebook/wav2vec2-large-lv60" \
	--dataset_config_name="clean" \
	--train_split_name="train" \
	--eval_split_name="test" \
	--output_dir="$CACHE_PATH/wav2vec2-librispeech-clean-100h-demo-dist" \
	--preprocessing_num_workers="16" \
	--overwrite_output_dir \
	--num_train_epochs="1" \
	--per_device_train_batch_size="4" \
	--gradient_accumulation_steps="1" \
	--learning_rate="3e-4" \
	--warmup_steps="5" \
	--eval_strategy="steps" \
	--text_column_name="text" \
	--save_steps="4" \
	--eval_steps="1" \
	--logging_steps="1" \
	--layerdrop="0.0" \
	--save_total_limit="3" \
	--freeze_feature_extractor \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
	--fp16 \
	--group_by_length \
	--do_eval \
	--online
