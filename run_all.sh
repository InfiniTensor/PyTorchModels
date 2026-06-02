#!/bin/bash

# ==============================================================================
#                     MX-01 模型能力测评脚本
#                训练+推理性能测试 (80 models, 10 domains)
#
#   用法:
#     ./run_all.sh                          # 全部训练+推理
#     ./run_all.sh all                      # 全部训练+推理
#     ./run_all.sh train                    # 全部仅训练
#     ./run_all.sh eval                     # 全部仅推理
#     ./run_all.sh train Segmentation       # 仅 Segmentation 训练
#     ./run_all.sh eval Detection NLP       # 仅 Detection+NLP 推理
#     ./run_all.sh all ImageClassification  # ImageClassification 训练+推理
#
#   支持的测试组:
#     Detection  ImageClassification  GAN  NLP  RL  Recommendation
#     SR  Segmentation  Speech  TimeSeriesPrediction
#
# ==============================================================================

set -o pipefail

# --- 加载平台环境 ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ -f env.sh ]; then
    source env.sh
fi
PLATFORM_ENV=${PLATFORM_ENV:-"UNKNOWN"}

# 确保 CUDA_VISIBLE_DEVICES 生效（由 env.sh 或命令行设置）
export CUDA_VISIBLE_DEVICES
echo -e "${COLOR_CYAN}Using GPUs: ${CUDA_VISIBLE_DEVICES}${COLOR_NC}"

# --- 颜色定义 ---
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'
COLOR_NC='\033[0m'

# --- 超时配置 ---
TRAIN_TIMEOUT="10m"
EVAL_TIMEOUT="5m"

# --- 日志目录 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/run_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# --- 当前日期 ---
REPORT_DATE=$(date +%Y-%m-%d)

# --- ImageClassification模型列表 ---
IC_MODELS=(
    alexnet convnext_base convnext_large convnext_small convnext_tiny
    densenet121 densenet161 densenet169 densenet201
    efficientnet_b0 efficientnet_b1 efficientnet_b2 efficientnet_b3
    efficientnet_b4 efficientnet_b5 efficientnet_b6 efficientnet_b7
    googlenet inception_v3 mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3
    mobilenet_v2 mobilenet_v3_large mobilenet_v3_small
    regnet_x_16gf regnet_x_1_6gf regnet_x_32gf regnet_x_3_2gf
    regnet_x_400mf regnet_x_800mf regnet_x_8gf regnet_y_128gf
    regnet_y_16gf regnet_y_1_6gf regnet_y_32gf regnet_y_3_2gf
    regnet_y_400mf regnet_y_800mf regnet_y_8gf resnet101 resnet152
    resnet18 resnet34 resnet50 resnext101_32x8d resnext50_32x4d
    shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0
    squeezenet1_0 squeezenet1_1 vgg11 vgg11_bn vgg13 vgg13_bn vgg16
    vgg16_bn vgg19 vgg19_bn vit_b_16 vit_b_32 vit_l_16 vit_l_32
    wide_resnet101_2 wide_resnet50_2
)
# IC_MODELS 子集（调试用）：
#   resnet18 mobilenet_v2 vgg16 inception_v3 densenet121 squeezenet1_0 efficientnet_b0 shufflenet_v2_x1_0

# --- 结果存储 ---
declare -a JSON_RESULTS   # JSON数组元素
TOTAL_MODELS=0
TRAIN_OK=0
TRAIN_FAIL=0
EVAL_OK=0
EVAL_FAIL=0

# ==============================================================================
#                           指标提取函数
# ==============================================================================

# 从日志中提取训练吞吐量 (samples/s)
extract_train_throughput() {
    local logfile="$1"
    # ImageClassification: "Train throughput: XXX samples/s"
    local val=$(grep -oP 'Train throughput:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # 通用: "Throughput: XXX samples/s"
    val=$(grep -oP 'Throughput:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # HuggingFace Trainer: "train_samples_per_second = XXX"
    val=$(grep -oP 'train_samples_per_second\s*=\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # Detection/Segmentation: "Batch Time X.XXX (Y.YYY)" 从平均耗时推算吞吐
    local batch_time=$(grep -oP 'Batch Time [\d.]+ \(\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$batch_time" ]; then
        echo "scale=2; 1/$batch_time" | bc 2>/dev/null
        return
    fi
    # Segmentation: "Avg it/s: XX.XX"
    val=$(grep -oP 'Avg it/s:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # ImageClassification PyTorch: "Time  X.XXX ( Y.YYY)" 括号内是平均 batch 耗时，batch_size=64
    local ic_time=$(grep -oP 'Time\s+[\d.]+\s+\(\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$ic_time" ]; then
        echo "scale=2; 64/$ic_time" | bc 2>/dev/null
        return
    fi
    echo ""
}

# 从日志中提取单步耗时 (ms/step)
extract_step_time() {
    local logfile="$1"
    # Detection/Segmentation: "Batch Time X.XXX (Y.YYY)" 平均耗时(秒)转毫秒
    local val=$(grep -oP 'Batch Time [\d.]+ \(\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then
        echo "scale=2; $val * 1000" | bc 2>/dev/null
        return
    fi
    # ImageClassification: "Time.*XXX ms"
    val=$(grep -oP 'Batch time.*?[\d.]+\s*ms' "$logfile" 2>/dev/null | grep -oP '[\d.]+' | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # Segmentation: "Avg it/s: XX.XX" 转毫秒
    val=$(grep -oP 'Avg it/s:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then
        echo "scale=2; 1000/$val" | bc 2>/dev/null
        return
    fi
    # ImageClassification PyTorch: "Time  X.XXX ( Y.YYY)" 括号内是秒/batch，转毫秒
    val=$(grep -oP 'Time\s+[\d.]+\s+\(\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then
        echo "scale=2; $val * 1000" | bc 2>/dev/null
        return
    fi
    echo ""
}

# 从日志中提取推理吞吐量 (samples/s)
extract_eval_throughput() {
    local logfile="$1"
    # 通用: "Inference throughput: XXX samples/s"
    local val=$(grep -oP 'Inference throughput:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # ImageClassification: "Evaluate throughput.*XXX samples/s"
    val=$(grep -oP 'Evaluate throughput.*?\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # HuggingFace Trainer: "eval_samples_per_second = XXX"
    val=$(grep -oP 'eval_samples_per_second\s*=\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # images/s 变体
    val=$(grep -oP 'Inference throughput:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # images/s
    val=$(grep -oP '[\d.]+(?=\s*images?/s)' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # ImageClassification PyTorch: "Time  X.XXX ( Y.YYY)" 括号内是平均 batch 耗时，batch_size=64
    local ic_time=$(grep -oP 'Time\s+[\d.]+\s+\(\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$ic_time" ]; then
        echo "scale=2; 64/$ic_time" | bc 2>/dev/null
        return
    fi
    echo ""
}

# 从日志中提取推理平均时延 (ms)
extract_eval_latency() {
    local logfile="$1"
    local val=$(grep -oP 'Average inference latency:\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # HuggingFace Trainer: "eval_runtime = X:XX:XX.xx" -> convert to ms/sample
    local eval_runtime=$(grep -oP 'eval_runtime\s*=\s*\K[\d:.]+' "$logfile" 2>/dev/null | tail -1)
    local eval_samples=$(grep -oP 'eval_samples\s*=\s*\K[\d]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$eval_runtime" ] && [ -n "$eval_samples" ]; then
        # eval_runtime is like "0:00:18.47" or just seconds
        local seconds=$(echo "$eval_runtime" | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else print $1}')
        if [ -n "$seconds" ] && [ "$eval_samples" -gt 0 ] 2>/dev/null; then
            echo "scale=2; $seconds * 1000 / $eval_samples" | bc 2>/dev/null
            return
        fi
    fi
    # ms/image 变体
    val=$(grep -oP '[\d.]+(?=\s*ms/)' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then echo "$val"; return; fi
    # ImageClassification PyTorch: "Time  X.XXX ( Y.YYY)" 括号内是秒/batch，转毫秒
    val=$(grep -oP 'Time\s+[\d.]+\s+\(\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1)
    if [ -n "$val" ]; then
        echo "scale=2; $val * 1000" | bc 2>/dev/null
        return
    fi
    echo ""
}

# 提取精度指标
extract_metric() {
    local logfile="$1"
    local domain="$2"
    case "$domain" in
        Detection)
            grep -oP 'mAP[):\s]*\K[\d.]+' "$logfile" 2>/dev/null | tail -1
            ;;
        ImageClassification)
            grep -oP 'Acc@1\s+\K[\d.]+' "$logfile" 2>/dev/null | tail -1
            ;;
        Segmentation)
            grep -oP 'mIoU[:\s]*\K[\d.]+' "$logfile" 2>/dev/null | tail -1
            ;;
        Speech)
            grep -oP 'cer[：:\s]+\K[\d.]+' "$logfile" 2>/dev/null | tail -1
            ;;
        TimeSeriesPrediction)
            grep -oP 'Acc[=:]\s*\K[\d.]+' "$logfile" 2>/dev/null | tail -1
            ;;
        *)
            grep -oP '(?:accuracy|Acc@1|mAP|mIoU|F1|Loss)[:\s=]+\K[\d.]+' "$logfile" 2>/dev/null | tail -1
            ;;
    esac
}

# ==============================================================================
#                           JSON 辅助函数
# ==============================================================================

# 生成单个模型结果的JSON片段
make_result_json() {
    local domain="$1" model="$2"
    local train_status="$3" train_tput="$4" train_step="$5"
    local eval_status="$6" eval_tput="$7" eval_lat="$8" eval_metric="$9"

    local train_tput_json="null"
    local train_step_json="null"
    local eval_tput_json="null"
    local eval_lat_json="null"
    local eval_metric_json="null"

    [ -n "$train_tput" ] && train_tput_json="$train_tput"
    [ -n "$train_step" ] && train_step_json="$train_step"
    [ -n "$eval_tput" ] && eval_tput_json="$eval_tput"
    [ -n "$eval_lat" ] && eval_lat_json="$eval_lat"
    [ -n "$eval_metric" ] && eval_metric_json="\"$eval_metric\""

    cat <<JSONEOF
    {
      "domain": "$domain",
      "model": "$model",
      "train": {"status": "$train_status", "throughput_sps": $train_tput_json, "step_time_ms": $train_step_json},
      "eval": {"status": "$eval_status", "throughput_sps": $eval_tput_json, "latency_ms": $eval_lat_json, "metric": $eval_metric_json}
    }
JSONEOF
}

# ==============================================================================
#                           任务执行函数
# ==============================================================================

# 运行单个任务（带超时），返回退出码
# 用法: run_task logfile timeout command [args...]
run_task() {
    local logfile="$1"; shift
    local timeout_val="$1"; shift
    local task_name="$1"; shift

    if [ "$timeout_val" = "none" ]; then
        echo -e "${COLOR_CYAN}  [RUN] $task_name (no timeout)${COLOR_NC}"
        bash -c "$@" > "$logfile" 2>&1
        return $?
    else
        echo -e "${COLOR_CYAN}  [RUN] $task_name (timeout: ${timeout_val})${COLOR_NC}"
        timeout "$timeout_val" bash -c "$@" > "$logfile" 2>&1
        return $?
    fi
}

# ==============================================================================
#                      各领域模型执行函数
# ==============================================================================

# --- Detection 模型 ---
run_detection_train() {
    local model="$1"
    local logfile="$2"
    case "$model" in
        fasterrcnn)
            run_task "$logfile" "$TRAIN_TIMEOUT" "Detection/fasterrcnn train" \
                'cd Detection/fasterrcnn && DATA_DIR=../data/VOCdevkit bash run_train.sh'
            ;;
        ssd)
            run_task "$logfile" "$TRAIN_TIMEOUT" "Detection/ssd train" \
                'cd Detection/ssd && DATA_DIR=../data/VOCdevkit bash run_train.sh'
            ;;
        yolo)
            run_task "$logfile" "$TRAIN_TIMEOUT" "Detection/yolo train" \
                'cd Detection/yolo && MODEL=yolov5s DATA_DIR=../data/coco bash run_train.sh'
            ;;
    esac
}

run_detection_eval() {
    local model="$1"
    local logfile="$2"
    case "$model" in
        fasterrcnn)
            run_task "$logfile" "$EVAL_TIMEOUT" "Detection/fasterrcnn eval" \
                'cd Detection/fasterrcnn && DATA_DIR=../data/VOCdevkit CKPT_DIR=./ bash run_eval.sh'
            ;;
        ssd)
            run_task "$logfile" "$EVAL_TIMEOUT" "Detection/ssd eval" \
                'cd Detection/ssd && DATA_DIR=../data/VOCdevkit bash run_eval.sh'
            ;;
        yolo)
            run_task "$logfile" "$EVAL_TIMEOUT" "Detection/yolo eval" \
                'cd Detection/yolo && MODEL=yolov5s DATA_DIR=../data/coco bash run_eval.sh'
            ;;
    esac
}

# --- ImageClassification 模型 ---
run_ic_train() {
    local model="$1"
    local logfile="$2"
    run_task "$logfile" "none" "ImageClassification/$model train" \
        "cd ImageClassification/TorchVision && DATA_DIR=../data/imagenet2012 ARCH=$model python main.py -a $model --gpu 0 --dummy --batch-size 64 ../data/imagenet2012"
}

run_ic_eval() {
    local model="$1"
    local logfile="$2"
    run_task "$logfile" "$EVAL_TIMEOUT" "ImageClassification/$model eval" \
        "cd ImageClassification/TorchVision && DATA_DIR=../data/imagenet2012 ARCH=$model bash run_eval.sh"
}

# --- ImageClassification 批量训练 ---
# IC_MODEL_TIMEOUT: 每个模型训练时间（秒），默认 300（5分钟）
IC_MODEL_TIMEOUT=${IC_MODEL_TIMEOUT:-300}
run_ic_batch_train() {
    local logfile="$1"
    local model_list="${IC_MODELS[*]}"
    # 不设总超时，由脚本内每个模型单独控制超时
    run_task "$logfile" "none" "ImageClassification/batch_train" \
        "cd ImageClassification/TorchVision && DATA_DIR=../data/imagenet2012 IC_MODEL_TIMEOUT=$IC_MODEL_TIMEOUT IC_MODELS='$model_list' bash run_all_models_train.sh"
}

# --- ImageClassification 批量推理 ---
run_ic_batch_eval() {
    local logfile="$1"
    local model_list="${IC_MODELS[*]}"
    # 不设总超时，由脚本内每个模型单独控制超时
    run_task "$logfile" "none" "ImageClassification/batch_eval" \
        "cd ImageClassification/TorchVision && DATA_DIR=../data/imagenet2012 IC_MODEL_TIMEOUT=$IC_MODEL_TIMEOUT IC_MODELS='$model_list' bash run_all_models_eval.sh"
}

# --- GAN ---
run_gan_train() {
    local logfile="$1"
    run_task "$logfile" "$TRAIN_TIMEOUT" "GAN/dcgan train" \
        'cd GAN/dcgan && DATA_DIR=../data/lsun bash run_train.sh'
}

run_gan_eval() {
    local logfile="$1"
    # DCGAN 使用 train.py --dry-run 做推理基准测试
    run_task "$logfile" "$EVAL_TIMEOUT" "GAN/dcgan eval" \
        'cd GAN/dcgan && PYTHONUNBUFFERED=1 python train.py --dataset fake --cuda --dry-run'
}

# --- NLP ---
run_nlp_train() {
    local logfile="$1"
    run_task "$logfile" "20m" "NLP/HuggingFace train" \
        'cd NLP/HuggingFace && bash run_train_online.sh'
}

run_nlp_eval() {
    local logfile="$1"
    run_task "$logfile" "$EVAL_TIMEOUT" "NLP/HuggingFace eval" \
        'cd NLP/HuggingFace && bash run_eval_online.sh'
}

# --- RL ---
run_rl_train() {
    local logfile="$1"
    run_task "$logfile" "$TRAIN_TIMEOUT" "RL/dqn train" \
        'cd RL/dqn && bash run_train.sh checkpoints 100 0.0001'
}

run_rl_eval() {
    local logfile="$1"
    run_task "$logfile" "$EVAL_TIMEOUT" "RL/dqn eval" \
        'cd RL/dqn && bash run_eval.sh ./checkpoints/20.pth'
}

# --- Recommendation ---
run_rec_train() {
    local logfile="$1"
    run_task "$logfile" "$TRAIN_TIMEOUT" "Recommendation/DLRM train" \
        'cd Recommendation/DLRM && DATA_DIR=../data/ml-20mx4x16 bash run_train.sh'
}

run_rec_eval() {
    local logfile="$1"
    run_task "$logfile" "$EVAL_TIMEOUT" "Recommendation/DLRM eval" \
        'cd Recommendation/DLRM && DATA_DIR=../data MODEL=./checkpoints/dlrmamp_0.pth bash run_eval.sh'
}

# --- SR ---
run_sr_train() {
    local logfile="$1"
    run_task "$logfile" "$TRAIN_TIMEOUT" "SR/ESPCN train" \
        'cd SR/ESPCN && bash run_train.sh'
}

run_sr_eval() {
    local logfile="$1"
    run_task "$logfile" "$EVAL_TIMEOUT" "SR/ESPCN eval" \
        'cd SR/ESPCN && bash run_eval.sh'
}

# --- Segmentation ---
run_seg_train() {
    local model="$1"
    local logfile="$2"
    run_task "$logfile" "$TRAIN_TIMEOUT" "Segmentation/$model train" \
        "cd Segmentation/$model && bash run_train.sh"
}

run_seg_eval() {
    local model="$1"
    local logfile="$2"
    run_task "$logfile" "$EVAL_TIMEOUT" "Segmentation/$model eval" \
        "cd Segmentation/$model && bash run_eval.sh"
}

# --- Speech ---
run_speech_train() {
    local model="$1"
    local logfile="$2"
    case "$model" in
        deepspeech2)
            run_task "$logfile" "$TRAIN_TIMEOUT" "Speech/deepspeech2 train" \
                'cd Speech/deepspeech2 && bash run_train.sh'
            ;;
        wav2vec)
            run_task "$logfile" "$TRAIN_TIMEOUT" "Speech/wav2vec train" \
                'cd Speech/wav2vec && bash run_train_online.sh'
            ;;
    esac
}

run_speech_eval() {
    local model="$1"
    local logfile="$2"
    case "$model" in
        deepspeech2)
            run_task "$logfile" "$EVAL_TIMEOUT" "Speech/deepspeech2 eval" \
                'cd Speech/deepspeech2 && bash run_eval.sh'
            ;;
        wav2vec)
            run_task "$logfile" "$EVAL_TIMEOUT" "Speech/wav2vec eval" \
                'cd Speech/wav2vec && bash run_eval_online.sh'
            ;;
    esac
}

# --- TimeSeriesPrediction ---
run_ts_train() {
    local model="$1"
    local logfile="$2"
    case "$model" in
        lstm)
            run_task "$logfile" "$TRAIN_TIMEOUT" "TimeSeriesPrediction/lstm train" \
                'cd TimeSeriesPrediction/lstm && bash run_train.sh ../data/complete_data.csv 200 512 0.0001'
            ;;
        tcn)
            run_task "$logfile" "$TRAIN_TIMEOUT" "TimeSeriesPrediction/tcn train" \
                'cd TimeSeriesPrediction/tcn && bash run_train_val.sh'
            ;;
    esac
}

run_ts_eval() {
    local model="$1"
    local logfile="$2"
    case "$model" in
        lstm)
            run_task "$logfile" "$EVAL_TIMEOUT" "TimeSeriesPrediction/lstm eval" \
                'cd TimeSeriesPrediction/lstm && bash run_eval.sh ../data/complete_data.csv ./checkpoints/lstm_best.pt'
            ;;
        tcn)
            run_task "$logfile" "$EVAL_TIMEOUT" "TimeSeriesPrediction/tcn eval" \
                'cd TimeSeriesPrediction/tcn && bash run_eval.sh'
            ;;
    esac
}

# ==============================================================================
#                       模型定义表 (domain -> models)
# ==============================================================================

# 领域中文名映射
declare -A DOMAIN_CN
DOMAIN_CN[Detection]="目标检测"
DOMAIN_CN[ImageClassification]="图像分类"
DOMAIN_CN[GAN]="对抗生成"
DOMAIN_CN[NLP]="NLP"
DOMAIN_CN[RL]="强化学习"
DOMAIN_CN[Recommendation]="推荐"
DOMAIN_CN[SR]="超分辨率"
DOMAIN_CN[Segmentation]="语义分割"
DOMAIN_CN[Speech]="语音识别"
DOMAIN_CN[TimeSeriesPrediction]="时序预测"

# 各领域模型列表
declare -A DOMAIN_MODELS
DOMAIN_MODELS[Detection]="fasterrcnn ssd yolo"
DOMAIN_MODELS[ImageClassification]="__batch__"  # 特殊标记: 64个模型批量处理
DOMAIN_MODELS[GAN]="dcgan"
DOMAIN_MODELS[NLP]="bert"
DOMAIN_MODELS[RL]="dqn"
DOMAIN_MODELS[Recommendation]="dlrm"
DOMAIN_MODELS[SR]="espcn"
DOMAIN_MODELS[Segmentation]="deeplab fcn lraspp unet"
DOMAIN_MODELS[Speech]="deepspeech2 wav2vec"
DOMAIN_MODELS[TimeSeriesPrediction]="lstm tcn"

# 有效领域列表
ALL_DOMAINS=(Detection ImageClassification GAN NLP RL Recommendation SR Segmentation Speech TimeSeriesPrediction)

# ==============================================================================
#                       运行单个模型（训练+推理）
# ==============================================================================

process_model() {
    local domain="$1"
    local model="$2"
    local mode="$3"  # all, train, eval

    local train_status="SKIP"
    local eval_status="SKIP"
    local train_tput="" train_step="" eval_tput="" eval_lat="" eval_metric=""

    TOTAL_MODELS=$((TOTAL_MODELS + 1))

    echo -e "\n${COLOR_BLUE}--- [$domain] $model ---${COLOR_NC}"

    # --- 训练 ---
    if [ "$mode" = "all" ] || [ "$mode" = "train" ]; then
        local train_log="${LOG_DIR}/${domain}_${model}_train.log"
        local rc=0

        case "$domain" in
            Detection)       run_detection_train "$model" "$train_log" || rc=$? ;;
            ImageClassification) run_ic_train "$model" "$train_log" || rc=$? ;;
            GAN)             run_gan_train "$train_log" || rc=$? ;;
            NLP)             run_nlp_train "$train_log" || rc=$? ;;
            RL)              run_rl_train "$train_log" || rc=$? ;;
            Recommendation)  run_rec_train "$train_log" || rc=$? ;;
            SR)              run_sr_train "$train_log" || rc=$? ;;
            Segmentation)    run_seg_train "$model" "$train_log" || rc=$? ;;
            Speech)          run_speech_train "$model" "$train_log" || rc=$? ;;
            TimeSeriesPrediction) run_ts_train "$model" "$train_log" || rc=$? ;;
        esac

        if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
            train_status="OK"
            [ $rc -eq 124 ] && train_status="OK(timeout)"
            TRAIN_OK=$((TRAIN_OK + 1))
            train_tput=$(extract_train_throughput "$train_log")
            train_step=$(extract_step_time "$train_log")
            echo -e "  ${COLOR_GREEN}TRAIN: $train_status${COLOR_NC} throughput=${train_tput:-N/A} step_time=${train_step:-N/A}"
        else
            train_status="FAIL"
            TRAIN_FAIL=$((TRAIN_FAIL + 1))
            echo -e "  ${COLOR_RED}TRAIN: FAIL (rc=$rc)${COLOR_NC}"
        fi
    fi

    # --- 推理 ---
    if [ "$mode" = "all" ] || [ "$mode" = "eval" ]; then
        local eval_log="${LOG_DIR}/${domain}_${model}_eval.log"
        local rc=0

        case "$domain" in
            Detection)       run_detection_eval "$model" "$eval_log" || rc=$? ;;
            ImageClassification) run_ic_eval "$model" "$eval_log" || rc=$? ;;
            GAN)             run_gan_eval "$eval_log" || rc=$? ;;
            NLP)             run_nlp_eval "$eval_log" || rc=$? ;;
            RL)              run_rl_eval "$eval_log" || rc=$? ;;
            Recommendation)  run_rec_eval "$eval_log" || rc=$? ;;
            SR)              run_sr_eval "$eval_log" || rc=$? ;;
            Segmentation)    run_seg_eval "$model" "$eval_log" || rc=$? ;;
            Speech)          run_speech_eval "$model" "$eval_log" || rc=$? ;;
            TimeSeriesPrediction) run_ts_eval "$model" "$eval_log" || rc=$? ;;
        esac

        if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
            eval_status="OK"
            [ $rc -eq 124 ] && eval_status="OK(timeout)"
            EVAL_OK=$((EVAL_OK + 1))
            eval_tput=$(extract_eval_throughput "$eval_log")
            eval_lat=$(extract_eval_latency "$eval_log")
            eval_metric=$(extract_metric "$eval_log" "$domain")
            echo -e "  ${COLOR_GREEN}EVAL: $eval_status${COLOR_NC} throughput=${eval_tput:-N/A} latency=${eval_lat:-N/A} metric=${eval_metric:-N/A}"
        else
            eval_status="FAIL"
            EVAL_FAIL=$((EVAL_FAIL + 1))
            echo -e "  ${COLOR_RED}EVAL: FAIL (rc=$rc)${COLOR_NC}"
        fi
    fi

    # 记录JSON结果
    local metric_val=""
    [ -n "$eval_metric" ] && metric_val="Acc@1 $eval_metric"
    JSON_RESULTS+=("$(make_result_json "$domain" "$model" "$train_status" "$train_tput" "$train_step" "$eval_status" "$eval_tput" "$eval_lat" "$metric_val")")
}

# ==============================================================================
#                  ImageClassification 批量处理 (64个模型)
# ==============================================================================

process_ic_batch() {
    local mode="$1"

    echo -e "\n${COLOR_BLUE}========== ImageClassification 批量处理 (${#IC_MODELS[@]} models) ==========${COLOR_NC}"

    if [ "$mode" = "all" ] || [ "$mode" = "train" ]; then
        local batch_train_log="${LOG_DIR}/ImageClassification_batch_train.log"
        local rc=0

        echo -e "${COLOR_CYAN}  [BATCH TRAIN] ${#IC_MODELS[@]} models via run_all_models_train.sh${COLOR_NC}"
        run_ic_batch_train "$batch_train_log" || rc=$?

        # 解析批量训练日志，按模型拆分
        local current_model=""
        local model_log=""
        for model in "${IC_MODELS[@]}"; do
            TOTAL_MODELS=$((TOTAL_MODELS + 1))
            # 提取该模型的日志片段
            local model_train_log="${LOG_DIR}/ImageClassification_${model}_train.log"
            # 从批量日志中提取该模型对应的行
            awk "/Training ${model} start/,/Training ${model} finish/" "$batch_train_log" > "$model_train_log" 2>/dev/null

            local train_tput="" train_step=""
            if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
                train_tput=$(extract_train_throughput "$model_train_log")
                train_step=$(extract_step_time "$model_train_log")
                TRAIN_OK=$((TRAIN_OK + 1))
                echo -e "  ${COLOR_GREEN}$model TRAIN: OK${COLOR_NC} throughput=${train_tput:-N/A}"
                JSON_RESULTS+=("$(make_result_json "ImageClassification" "$model" "OK" "$train_tput" "$train_step" "SKIP" "" "" "")")
            else
                TRAIN_FAIL=$((TRAIN_FAIL + 1))
                echo -e "  ${COLOR_RED}$model TRAIN: FAIL${COLOR_NC}"
                JSON_RESULTS+=("$(make_result_json "ImageClassification" "$model" "FAIL" "" "" "SKIP" "" "" "")")
            fi
        done
    fi

    if [ "$mode" = "all" ] || [ "$mode" = "eval" ]; then
        local batch_eval_log="${LOG_DIR}/ImageClassification_batch_eval.log"
        local rc=0

        echo -e "${COLOR_CYAN}  [BATCH EVAL] ${#IC_MODELS[@]} models via run_all_models_eval.sh${COLOR_NC}"
        run_ic_batch_eval "$batch_eval_log" || rc=$?

        for model in "${IC_MODELS[@]}"; do
            if [ "$mode" = "eval" ]; then
                TOTAL_MODELS=$((TOTAL_MODELS + 1))
            fi
            local model_eval_log="${LOG_DIR}/ImageClassification_${model}_eval.log"
            awk "/Evaluating ${model} start/,/Evaluating ${model} finish/" "$batch_eval_log" > "$model_eval_log" 2>/dev/null

            local eval_tput="" eval_lat="" eval_metric=""
            if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then
                eval_tput=$(extract_eval_throughput "$model_eval_log")
                eval_lat=$(extract_eval_latency "$model_eval_log")
                eval_metric=$(extract_metric "$model_eval_log" "ImageClassification")
                EVAL_OK=$((EVAL_OK + 1))
                echo -e "  ${COLOR_GREEN}$model EVAL: OK${COLOR_NC} throughput=${eval_tput:-N/A} Acc@1=${eval_metric:-N/A}"

                # 更新对应的JSON结果
                local idx=0
                local dq='"'
                for r in "${JSON_RESULTS[@]}"; do
                    if echo "$r" | grep -q "${dq}model${dq}: ${dq}${model}${dq}"; then
                        local t_status=$(echo "$r" | grep -oP "${dq}train${dq}:.*?${dq}status${dq}: ${dq}\K[^${dq}]*")
                        local t_tput=$(echo "$r" | grep -oP "throughput_sps${dq}: \K[^,}]*")
                        local t_step=$(echo "$r" | grep -oP "step_time_ms${dq}: \K[^,}]*")
                        JSON_RESULTS[$idx]=$(make_result_json "ImageClassification" "$model" \
                            "$t_status" "$t_tput" "$t_step" \
                            "OK" "$eval_tput" "$eval_lat" "Acc@1 $eval_metric")
                        break
                    fi
                    idx=$((idx + 1))
                done
            else
                EVAL_FAIL=$((EVAL_FAIL + 1))
                echo -e "  ${COLOR_RED}$model EVAL: FAIL${COLOR_NC}"
            fi
        done
    fi
}

# ==============================================================================
#                           报告生成
# ==============================================================================

generate_report() {
    local json_file="${SCRIPT_DIR}/benchmark_report_${TIMESTAMP}.json"

    # --- JSON 报告 ---
    {
        echo "{"
        echo "  \"platform\": \"${PLATFORM_ENV}\","
        echo "  \"date\": \"${REPORT_DATE}\","
        echo "  \"summary\": {"
        echo "    \"total_models\": ${TOTAL_MODELS},"
        echo "    \"train_passed\": ${TRAIN_OK},"
        echo "    \"train_failed\": ${TRAIN_FAIL},"
        echo "    \"eval_passed\": ${EVAL_OK},"
        echo "    \"eval_failed\": ${EVAL_FAIL},"
        local train_total=$((TRAIN_OK + TRAIN_FAIL))
        local eval_total=$((EVAL_OK + EVAL_FAIL))
        if [ $train_total -gt 0 ]; then
            echo "    \"train_pass_rate\": $(echo "scale=3; $TRAIN_OK / $train_total" | bc),"
        else
            echo "    \"train_pass_rate\": null,"
        fi
        if [ $eval_total -gt 0 ]; then
            echo "    \"eval_pass_rate\": $(echo "scale=3; $EVAL_OK / $eval_total" | bc),"
        else
            echo "    \"eval_pass_rate\": null,"
        fi

        # 领域列表
        local first=1
        echo "    \"domains\": ["
        for d in "${ALL_DOMAINS[@]}"; do
            local models="${DOMAIN_MODELS[$d]}"
            local cnt=0
            if [ "$d" = "ImageClassification" ]; then
                cnt=${#IC_MODELS[@]}
            else
                cnt=$(echo "$models" | wc -w)
            fi
            if [ $first -eq 1 ]; then first=0; else echo ","; fi
            printf '      "%s(%d)"' "${DOMAIN_CN[$d]}" "$cnt"
        done
        echo ""
        echo "    ]"
        echo "  },"
        echo "  \"results\": ["

        local first=1
        for r in "${JSON_RESULTS[@]}"; do
            if [ $first -eq 1 ]; then first=0; else echo ","; fi
            echo "$r"
        done
        echo ""
        echo "  ]"
        echo "}"
    } > "$json_file"

    echo -e "\n${COLOR_GREEN}JSON report saved to: $json_file${COLOR_NC}"

    # --- 控制台报告 ---
    echo ""
    echo "========================================================="
    echo "MX-01 模型能力测评报告"
    echo "平台: ${PLATFORM_ENV} | 日期: ${REPORT_DATE}"
    echo "========================================================="
    printf "%-10s %-20s %-9s %-9s %-12s %-10s %-12s %-10s\n" \
        "领域" "模型" "训练" "推理" "训练吞吐" "单步耗时" "推理吞吐" "推理时延"
    echo "------------------------------------------------------------------------------------------"

    local dq='"'
    for r in "${JSON_RESULTS[@]}"; do
        local domain=$(echo "$r" | grep -oP "${dq}domain${dq}: ${dq}\K[^${dq}]*")
        local model=$(echo "$r" | grep -oP "${dq}model${dq}: ${dq}\K[^${dq}]*")
        local t_status=$(echo "$r" | grep -oP "${dq}train${dq}:.*?${dq}status${dq}: ${dq}\K[^${dq}]*" | head -1)
        local e_status=$(echo "$r" | grep -oP "${dq}eval${dq}:.*?${dq}status${dq}: ${dq}\K[^${dq}]*" | head -1)
        local t_tput=$(echo "$r" | grep -oP "throughput_sps${dq}: \K[^,}]*" | head -1)
        local t_step=$(echo "$r" | grep -oP "step_time_ms${dq}: \K[^,}]*" | head -1)
        local e_tput=$(echo "$r" | grep -oP "throughput_sps${dq}: \K[^,}]*" | sed -n '2p')
        local e_lat=$(echo "$r" | grep -oP "latency_ms${dq}: \K[^,}]*" | head -1)

        [ "$t_tput" = "null" ] && t_tput=""
        [ "$t_step" = "null" ] && t_step=""
        [ "$e_tput" = "null" ] && e_tput=""
        [ "$e_lat" = "null" ] && e_lat=""

        # 缩短状态显示
        [ "${t_status:0:2}" = "OK" ] && t_status="OK"
        [ "${e_status:0:2}" = "OK" ] && e_status="OK"

        printf "%-10s %-20s %-9s %-9s %-12s %-10s %-12s %-10s\n" \
            "${DOMAIN_CN[$domain]:-$domain}" \
            "$model" \
            "${t_status:--}" \
            "${e_status:--}" \
            "${t_tput:--} s/s" \
            "${t_step:--} ms" \
            "${e_tput:--} s/s" \
            "${e_lat:--} ms"
    done

    echo "------------------------------------------------------------------------------------------"

    # 汇总
    local total_pass=$((TRAIN_OK + EVAL_OK))
    local total_fail=$((TRAIN_FAIL + EVAL_FAIL))
    local total_tests=$((total_pass + total_fail))
    local pass_rate="0.0"
    if [ $total_tests -gt 0 ]; then
        pass_rate=$(echo "scale=1; $total_pass * 100 / $total_tests" | bc)
    fi

    echo "总计: ${TOTAL_MODELS} 模型 | 训练通过: ${TRAIN_OK}/${TRAIN_OK}+${TRAIN_FAIL} | 推理通过: ${EVAL_OK}/${EVAL_OK}+${EVAL_FAIL} | 综合通过率: ${pass_rate}%"
    echo -n "领域覆盖: "
    local first=1
    for d in "${ALL_DOMAINS[@]}"; do
        local models="${DOMAIN_MODELS[$d]}"
        local cnt=0
        if [ "$d" = "ImageClassification" ]; then
            cnt=${#IC_MODELS[@]}
        else
            cnt=$(echo "$models" | wc -w)
        fi
        if [ $first -eq 1 ]; then first=0; else echo -n " "; fi
        echo -n "${DOMAIN_CN[$d]}(${cnt})"
    done
    echo ""
    echo "========================================================="
    echo -e "详细日志目录: ${LOG_DIR}"
}

# ==============================================================================
#                              主流程
# ==============================================================================

START_TIME=$(date +%s)

# --- 解析参数 ---
MODE="all"
declare -a SELECTED_DOMAINS

if [ $# -eq 0 ]; then
    MODE="all"
    SELECTED_DOMAINS=("${ALL_DOMAINS[@]}")
else
    first_arg="$1"
    case "$first_arg" in
        all|train|eval)
            MODE="$first_arg"
            shift
            ;;
        *)
            # 第一个参数不是模式，可能是领域名
            MODE="all"
            ;;
    esac

    if [ $# -eq 0 ]; then
        # 没有指定领域，运行全部
        SELECTED_DOMAINS=("${ALL_DOMAINS[@]}")
    else
        # 验证领域名
        for group in "$@"; do
            found=0
            for d in "${ALL_DOMAINS[@]}"; do
                if [ "$group" = "$d" ]; then
                    found=1
                    break
                fi
            done
            if [ $found -eq 1 ]; then
                SELECTED_DOMAINS+=("$group")
            else
                echo -e "${COLOR_RED}错误: 未知测试组 '$group'${COLOR_NC}"
                echo "可用测试组: ${ALL_DOMAINS[*]}"
                exit 1
            fi
        done
    fi
fi

echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"
echo -e "${COLOR_YELLOW}MX-01 模型能力测评${COLOR_NC}"
echo -e "${COLOR_YELLOW}平台: ${PLATFORM_ENV} | 模式: ${MODE} | 日期: ${REPORT_DATE}${COLOR_NC}"
echo -e "${COLOR_YELLOW}测试领域: ${SELECTED_DOMAINS[*]}${COLOR_NC}"
echo -e "${COLOR_YELLOW}日志目录: ${LOG_DIR}${COLOR_NC}"
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"

# --- 执行测试 ---

for domain in "${SELECTED_DOMAINS[@]}"; do
    echo -e "\n${COLOR_BLUE}========================================${COLOR_NC}"
    echo -e "${COLOR_BLUE}  领域: ${DOMAIN_CN[$domain]} ($domain)${COLOR_NC}"
    echo -e "${COLOR_BLUE}========================================${COLOR_NC}"

    if [ "$domain" = "ImageClassification" ]; then
        # ImageClassification 使用批量处理
        process_ic_batch "$MODE"
    else
        models="${DOMAIN_MODELS[$domain]}"
        for model in $models; do
            process_model "$domain" "$model" "$MODE"
        done
    fi
done

# --- 生成报告 ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "\n总耗时: ${DURATION} 秒"

generate_report

# --- 退出码 ---
if [ $TRAIN_FAIL -gt 0 ] || [ $EVAL_FAIL -gt 0 ]; then
    exit 1
fi
exit 0
