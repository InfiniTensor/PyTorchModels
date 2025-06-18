#!/bin/bash

# ==============================================================================
#                     模型训练自动化执行脚本
#
#   功能:
#   - 支持执行一个或多个指定的测试组。
#   - 修正了timeout无法执行bash函数的问题。
#   - 对指定任务应用超时限制 (10分钟)。
#   - 将“超时”视为成功状态进行统计，并在报告中特别注明。
#   - 在脚本文件所在位置创建唯一的日志目录。
#
#   用法:
#   - 执行所有测试: ./run_all_tests.sh
#   - 或:            ./run_all_tests.sh all
#   - 执行单个组:   ./run_all_tests.sh Detection
#   - 执行多个组:   ./run_all_tests.sh Detection Speech NLP
#
# ==============================================================================

# --- 配置区 ---
# 定义颜色，让输出更易读
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m' # 无颜色

# 定义超时时长
TIMEOUT_DURATION="10m" # 10分钟

# --- 初始化 ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/run_logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
echo "所有任务的详细日志将被保存在: $LOG_DIR"
echo "超时限制设置为: ${TIMEOUT_DURATION}"

# 初始化统计变量
declare -a SUCCESS_LIST
declare -a FAILURE_LIST
declare -a TIMEOUT_LIST
START_TIME=$(date +%s)

# --- 测试任务定义 ---
# 每个任务都在一个子Shell `( ... )` 中执行，以确保退出码准确且目录不相互影响。

run_test_fasterrcnn() {
    echo "-> 任务: Detection/fasterrcnn"
    ( cd Detection/fasterrcnn && DATA_DIR=../data/VOCdevkit bash run_train.sh )
}

run_test_ssd() {
    echo "-> 任务: Detection/ssd"
    ( cd Detection/ssd && DATA_DIR=../data/VOCdevkit bash run_train.sh )
}

run_test_yolo() {
    echo "-> 任务: Detection/yolo"
    ( cd Detection/yolo && MODEL=yolov5s DATA_DIR=../data/coco bash run_train.sh )
}

run_test_image_classification() {
    echo "-> 任务: ImageClassification/TorchVision"
    ( cd ImageClassification/TorchVision && DATA_DIR=../data/imagenet2012 bash run_all_models_train.sh )
}

run_test_gan() {
    echo "-> 任务: GAN/dcgan"
    ( cd GAN/dcgan && DATA_DIR=../data/lsun bash run_train.sh )
}

run_test_nlp() {
    echo "-> 任务: NLP/HuggingFace"
    ( cd NLP/HuggingFace && bash run_train_online.sh )
}

run_test_rl() {
    echo "-> 任务: RL/dqn"
    ( cd RL/dqn && bash run_train.sh checkpoints 100 0.0001 )
}

run_test_recommendation() {
    echo "-> 任务: Recommendation/DLRM"
    ( cd Recommendation/DLRM && DATA_DIR=../data/ml-20mx4x16 bash run_train.sh )
}

run_test_sr() {
    echo "-> 任务: SR/ESPCN"
    ( cd SR/ESPCN && bash run_train.sh )
}

run_test_segmentation_deeplab() {
    echo "-> 任务: Segmentation/deeplab"
    ( cd Segmentation/deeplab && bash run_train.sh )
}

run_test_segmentation_fcn() {
    echo "-> 任务: Segmentation/fcn"
    ( cd Segmentation/fcn && bash run_train.sh )
}

run_test_segmentation_lraspp() {
    echo "-> 任务: Segmentation/lraspp"
    ( cd Segmentation/lraspp && bash run_train.sh )
}

run_test_segmentation_unet() {
    echo "-> 任务: Segmentation/unet"
    ( cd Segmentation/unet && bash run_train.sh )
}

run_test_speech_deepspeech2() {
    echo "-> 任务: Speech/deepspeech2"
    ( cd Speech/deepspeech2 && bash run_train.sh )
}

run_test_speech_wav2vec() {
    echo "-> 任务: Speech/wav2vec"
    ( cd Speech/wav2vec && bash run_train_online.sh )
}

run_test_timeseries_lstm() {
    echo "-> 任务: TimeSeriesPrediction/lstm"
    ( cd TimeSeriesPrediction/lstm && bash run_train.sh ../data/complete_data.csv 200 512 0.0001 )
}

run_test_timeseries_tcn() {
    echo "-> 任务: TimeSeriesPrediction/tcn"
    ( cd TimeSeriesPrediction/tcn && bash run_train_val.sh )
}

# 将所有任务函数导出，以便子Shell (如timeout调用的) 可以访问它们
export -f run_test_fasterrcnn run_test_ssd run_test_yolo run_test_image_classification \
          run_test_gan run_test_nlp run_test_rl run_test_recommendation run_test_sr \
          run_test_segmentation_deeplab run_test_segmentation_fcn run_test_segmentation_lraspp \
          run_test_segmentation_unet run_test_speech_deepspeech2 run_test_speech_wav2vec \
          run_test_timeseries_lstm run_test_timeseries_tcn

# --- 任务分组定义 ---
GROUP_Detection=( run_test_fasterrcnn run_test_ssd run_test_yolo )
GROUP_ImageClassification=( run_test_image_classification )
GROUP_GAN=( run_test_gan )
GROUP_NLP=( run_test_nlp )
GROUP_RL=( run_test_rl )
GROUP_Recommendation=( run_test_recommendation )
GROUP_SR=( run_test_sr )
GROUP_Segmentation=( run_test_segmentation_deeplab run_test_segmentation_fcn run_test_segmentation_lraspp run_test_segmentation_unet )
GROUP_Speech=( run_test_speech_deepspeech2 run_test_speech_wav2vec )
GROUP_TimeSeriesPrediction=( run_test_timeseries_lstm run_test_timeseries_tcn )

ALL_TESTS=(
    "${GROUP_Detection[@]}" "${GROUP_ImageClassification[@]}" "${GROUP_GAN[@]}"
    "${GROUP_NLP[@]}" "${GROUP_RL[@]}" "${GROUP_Recommendation[@]}" "${GROUP_SR[@]}"
    "${GROUP_Segmentation[@]}" "${GROUP_Speech[@]}" "${GROUP_TimeSeriesPrediction[@]}"
)

# --- 核心执行逻辑 ---
execute() {
    local task_name=$1
    local task_log="${LOG_DIR}/${task_name}.log"
    local exit_code

    echo -e "\n${COLOR_BLUE}================== [ 开始执行: $task_name ] ==================${COLOR_NC}"
    echo "详细日志 -> ${task_log}"

    if [[ " ${GROUP_ImageClassification[@]} " =~ " ${task_name} " ]]; then
        echo "注意: 此任务没有超时限制。"
        bash -c "$task_name" > "$task_log" 2>&1
        exit_code=$?
    else
        timeout "$TIMEOUT_DURATION" bash -c "$task_name" > "$task_log" 2>&1
        exit_code=$?
    fi

    if [ $exit_code -eq 0 ]; then
        echo -e "${COLOR_GREEN}================== [ 成功 (正常完成): $task_name ] ==================${COLOR_NC}"
        SUCCESS_LIST+=("$task_name")
    elif [ $exit_code -eq 124 ]; then
        echo -e "${COLOR_GREEN}================== [ 成功 (超时终止): $task_name ] ==================${COLOR_NC}"
        SUCCESS_LIST+=("$task_name (超时)")
        TIMEOUT_LIST+=("$task_name")
    else
        echo -e "${COLOR_RED}================== [ 失败: $task_name | 退出码: $exit_code ] ==================${COLOR_NC}"
        echo -e "${COLOR_RED}失败详情已记录在: ${task_log}${COLOR_NC}"
        FAILURE_LIST+=("$task_name (退出码: $exit_code)")
    fi
}

# 根据用户输入决定执行哪些测试 (支持多个组)
declare -a TESTS_TO_RUN
declare -a INVALID_GROUPS

# 如果用户没有提供参数，或第一个参数是 'all'，则运行所有测试
if [ -z "$1" ] || [ "$1" == "all" ]; then
    echo -e "${COLOR_YELLOW}模式: 执行所有测试任务...${COLOR_NC}"
    TESTS_TO_RUN=("${ALL_TESTS[@]}")
else
    # 否则，遍历所有提供的参数
    echo -e "${COLOR_YELLOW}模式: 执行指定的测试组: $@${COLOR_NC}"
    for group in "$@"; do
        TARGET_GROUP_NAME="GROUP_$group"
        # 检查组名是否存在
        if (declare -p "$TARGET_GROUP_NAME" &>/dev/null); then
            # 如果存在，将其包含的任务添加到总列表中
            eval "TESTS_TO_RUN+=(\"\${${TARGET_GROUP_NAME}[@]}\")"
        else
            # 如果不存在，记录下来
            INVALID_GROUPS+=("$group")
        fi
    done
fi

# 检查是否有无效的组名
if [ ${#INVALID_GROUPS[@]} -gt 0 ]; then
    echo -e "\n${COLOR_RED}错误: 发现未知的测试组: ${INVALID_GROUPS[*]}${COLOR_NC}"
    echo "可用测试组: Detection, ImageClassification, GAN, NLP, RL, Recommendation, SR, Segmentation, Speech, TimeSeriesPrediction"
    exit 1
fi

# 对任务列表进行去重，以防用户重复输入同一个组名
UNIQUE_TESTS=($(printf "%s\n" "${TESTS_TO_RUN[@]}" | sort -u))

# 检查最终是否有任务需要执行
if [ ${#UNIQUE_TESTS[@]} -eq 0 ]; then
    echo -e "\n${COLOR_RED}错误: 没有有效的测试任务被选中。${COLOR_NC}"
    exit 1
fi

# 遍历并执行所有选定的测试任务
for task in "${UNIQUE_TESTS[@]}"; do
    execute "$task"
done


# --- 最终统计报告 ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
SUCCESS_COUNT=${#SUCCESS_LIST[@]}
FAILURE_COUNT=${#FAILURE_LIST[@]}
TIMEOUT_COUNT=${#TIMEOUT_LIST[@]}
TOTAL_COUNT=$((SUCCESS_COUNT + FAILURE_COUNT))

echo -e "\n\n${COLOR_BLUE}=======================================================${COLOR_NC}"
echo -e "${COLOR_YELLOW}                      测试执行总结                     ${COLOR_NC}"
echo -e "${COLOR_BLUE}=======================================================${COLOR_NC}"
echo "所有日志保存在目录: ${LOG_DIR}"
echo "总计执行任务: $TOTAL_COUNT"
echo "总耗时: ${DURATION} 秒"
echo -e "🟢 ${COLOR_GREEN}成功 (含超时): ${SUCCESS_COUNT} 个${COLOR_NC}"
echo -e "🔴 ${COLOR_RED}失败: ${FAILURE_COUNT} 个${COLOR_NC}"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo -e "\n${COLOR_RED}--- 失败任务列表 ---${COLOR_NC}"
    for item in "${FAILURE_LIST[@]}"; do
        echo "  - $item"
    done
fi

if [ $TIMEOUT_COUNT -gt 0 ]; then
    echo -e "\n${COLOR_YELLOW}--- 因超时而成功的任务列表 (${TIMEOUT_COUNT}个) ---${COLOR_NC}"
    for item in "${TIMEOUT_LIST[@]}"; do
        echo "  - $item"
    done
fi

echo -e "\n${COLOR_GREEN}--- 完整成功任务列表 (含正常完成及超时) ---${COLOR_NC}"
for item in "${SUCCESS_LIST[@]}"; do
    echo "  - $item"
done

echo -e "${COLOR_BLUE}=======================================================${COLOR_NC}"

# 如果有任何真正的失败，脚本以失败状态码退出
if [ $FAILURE_COUNT -gt 0 ]; then
    exit 1
fi

