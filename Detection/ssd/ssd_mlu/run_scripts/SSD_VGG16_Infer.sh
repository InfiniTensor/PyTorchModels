# GPU Eval Scripts
CUR_DIR=$(cd $(dirname $0);pwd)

export MLU_VISIBLE_DEVICES=1
#SAVE_PATH=$PROJ_DIR/data/output/eval/GPU_SSD_VGG16_eval
#if  [ ! -d $SAVE_PATH ];then
#    mkdir -p $SAVE_PATH
#fi

pushd $CUR_DIR/../models
# Eval model After Train based on last iteration
python -W ignore SSD_VGG16_test.py --trained_model $1 \
                         --voc_root $2 \
                         --device mlu

# /workspace/cv/detection/SSD_VGG16/ssd_final.pth /dataset
