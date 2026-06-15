#!/bin/bash

# ==============================================================================
#                     MX-01 分步测评脚本
#           按类别依次测试，每类独立日志，方便截图和复测
#
#   用法:
#     ./run_sequential.sh                              # 全部分步执行
#     ./run_sequential.sh train                        # 仅训练（分步）
#     ./run_sequential.sh eval                         # 仅推理（分步）
#     ./run_sequential.sh all 1 3                      # 跑第1、3组
#     ./run_sequential.sh train 4-6                    # 跑第4到6组（仅训练）
#     ./run_sequential.sh eval Detection Speech        # 按名称跑指定域
#     ./run_sequential.sh all Detection/fasterrcnn     # 跑单个模型
#     ./run_sequential.sh all det/yolo seg/unet ic/resnet18
#
#   指定组的方式（可混用）:
#     数字编号:    1 3 5                    第1、3、5组
#     范围:        2-4                      第2到4组
#     域名/简称:   Detection                按名称匹配
#     单模型:      Detection/fasterrcnn     加 /模型名 跑单个模型
#
#   测试分组:
#     1.  single     单模型合集 (GAN + NLP + RL + Recommendation + SR)   共 5 模型
#     2.  ts         TimeSeriesPrediction                                 共 2 模型
#     3.  speech     Speech                                               共 2 模型
#     4.  det        Detection                                            共 3 模型
#     5.  seg        Segmentation                                         共 4 模型
#     6.  ic1        图像分类(1/12)  alexnet convnext_tiny densenet121 densenet161 densenet169
#     7.  ic2        图像分类(2/12)  densenet201 efficientnet_b0 efficientnet_b1 efficientnet_b2 efficientnet_b3
#     8.  ic3        图像分类(3/12)  efficientnet_b4 efficientnet_b5 efficientnet_b6 googlenet inception_v3
#     9.  ic4        图像分类(4/12)  mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 mobilenet_v2
#     10. ic5        图像分类(5/12)  mobilenet_v3_large mobilenet_v3_small regnet_x_16gf regnet_x_1_6gf regnet_x_3_2gf
#     11. ic6        图像分类(6/12)  regnet_x_400mf regnet_x_800mf regnet_x_8gf regnet_y_16gf regnet_y_1_6gf
#     12. ic7        图像分类(7/12)  regnet_y_3_2gf regnet_y_400mf regnet_y_800mf regnet_y_8gf resnet101
#     13. ic8        图像分类(8/12)  resnet152 resnet18 resnet34 resnet50 resnext101_32x8d
#     14. ic9        图像分类(9/12)  resnext50_32x4d shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0
#     15. ic10       图像分类(10/12) squeezenet1_0 squeezenet1_1 vgg11 vgg11_bn vgg13
#     16. ic11       图像分类(11/12) vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
#     17. ic12       图像分类(12/12) vit_b_16 vit_b_32 vit_l_32 wide_resnet101_2 wide_resnet50_2
#
# ==============================================================================

set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

if [ -f env.sh ]; then
    source env.sh
fi
PLATFORM_ENV=${PLATFORM_ENV:-"UNKNOWN"}
export CUDA_VISIBLE_DEVICES

# --- 颜色 ---
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'
COLOR_NC='\033[0m'

# --- 测试分组定义 ---
# "分组简称|run_all.sh参数|中文名|环境变量(可选)"
declare -a TEST_GROUPS=(
    "single|GAN NLP RL Recommendation SR|单模型合集(GAN+NLP+RL+Rec+SR)|"
    "ts|TimeSeriesPrediction|时序预测|"
    "speech|Speech|语音识别|"
    "det|Detection|目标检测|"
    "seg|Segmentation|语义分割|"
    "ic1|ImageClassification|图像分类(1/12)|IC_MODELS='alexnet convnext_tiny densenet121 densenet161 densenet169'"
    "ic2|ImageClassification|图像分类(2/12)|IC_MODELS='densenet201 efficientnet_b0 efficientnet_b1 efficientnet_b2 efficientnet_b3'"
    "ic3|ImageClassification|图像分类(3/12)|IC_MODELS='efficientnet_b4 efficientnet_b5 efficientnet_b6 googlenet inception_v3'"
    "ic4|ImageClassification|图像分类(4/12)|IC_MODELS='mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 mobilenet_v2'"
    "ic5|ImageClassification|图像分类(5/12)|IC_MODELS='mobilenet_v3_large mobilenet_v3_small regnet_x_16gf regnet_x_1_6gf regnet_x_3_2gf'"
    "ic6|ImageClassification|图像分类(6/12)|IC_MODELS='regnet_x_400mf regnet_x_800mf regnet_x_8gf regnet_y_16gf regnet_y_1_6gf'"
    "ic7|ImageClassification|图像分类(7/12)|IC_MODELS='regnet_y_3_2gf regnet_y_400mf regnet_y_800mf regnet_y_8gf resnet101'"
    "ic8|ImageClassification|图像分类(8/12)|IC_MODELS='resnet152 resnet18 resnet34 resnet50 resnext101_32x8d'"
    "ic9|ImageClassification|图像分类(9/12)|IC_MODELS='resnext50_32x4d shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0'"
    "ic10|ImageClassification|图像分类(10/12)|IC_MODELS='squeezenet1_0 squeezenet1_1 vgg11 vgg11_bn vgg13'"
    "ic11|ImageClassification|图像分类(11/12)|IC_MODELS='vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn'"
    "ic12|ImageClassification|图像分类(12/12)|IC_MODELS='vit_b_16 vit_b_32 vit_l_32 wide_resnet101_2 wide_resnet50_2'"
)
TOTAL_DEF_GROUPS=${#TEST_GROUPS[@]}

# --- 参数解析 ---
MODE="${1:-all}"
case "$MODE" in
    all|train|eval) shift ;;
    *)
        # 第一个参数不是模式，默认 all
        MODE="all"
        ;;
esac

# --- 任务列表 ---
# 每项: "group_index" 或 "group_index:model_filter"
declare -a TASKS

usage_exit() {
    echo -e "${COLOR_RED}用法: $0 [all|train|eval] [分组...][/模型名]${COLOR_NC}"
    echo ""
    echo "  指定分组的方式（可混用）:"
    echo "    数字编号    1 3 5                  第1、3、5组"
    echo "    范围        2-4                    第2到4组"
    echo "    域名/简称   Detection              按名称匹配"
    echo "    单模型      Detection/fasterrcnn   加 /模型名 跑单个模型"
    echo ""
    echo "  可用分组:"
    local i=1
    for g in "${TEST_GROUPS[@]}"; do
        IFS='|' read -r short args cn env <<< "$g"
        printf "    %-3s %-22s %s\n" "$i." "$short" "$cn"
        i=$((i + 1))
    done
    exit 1
}

# 解析单个参数 → 添加到 TASKS
parse_arg() {
    local arg="$1"
    local model_filter=""
    local base_arg="$arg"

    # 提取 /model 后缀
    if [[ "$arg" == */* ]]; then
        base_arg="${arg%%/*}"
        model_filter="${arg#*/}"
        [ -z "$model_filter" ] && { echo -e "${COLOR_RED}错误: '$arg' 缺少模型名${COLOR_NC}"; usage_exit; }
    fi

    local group_idx=""

    # 1) 范围: 2-4
    if [[ "$base_arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        local s=${BASH_REMATCH[1]}
        local e=${BASH_REMATCH[2]}
        if [ "$s" -lt 1 ] || [ "$e" -gt "$TOTAL_DEF_GROUPS" ] || [ "$s" -gt "$e" ]; then
            echo -e "${COLOR_RED}错误: 无效范围 '$base_arg' (有效: 1-${TOTAL_DEF_GROUPS})${COLOR_NC}"
            usage_exit
        fi
        if [ -n "$model_filter" ]; then
            echo -e "${COLOR_RED}错误: 范围语法不支持指定模型${COLOR_NC}"
            usage_exit
        fi
        for ((i=s; i<=e; i++)); do
            TASKS+=("$((i - 1))")
        done
        return
    fi

    # 2) 数字编号: 1 3 5
    if [[ "$base_arg" =~ ^[0-9]+$ ]]; then
        if [ "$base_arg" -lt 1 ] || [ "$base_arg" -gt "$TOTAL_DEF_GROUPS" ]; then
            echo -e "${COLOR_RED}错误: 无效编号 '$base_arg' (有效: 1-${TOTAL_DEF_GROUPS})${COLOR_NC}"
            usage_exit
        fi
        group_idx=$((base_arg - 1))
    fi

    # 3) 名称/简称匹配
    if [ -z "$group_idx" ]; then
        for ((i=0; i<TOTAL_DEF_GROUPS; i++)); do
            IFS='|' read -r short args cn env <<< "${TEST_GROUPS[$i]}"
            if [ "$base_arg" = "$short" ] || echo "$args" | grep -qw "$base_arg"; then
                group_idx=$i
                break
            fi
        done
    fi

    if [ -z "$group_idx" ]; then
        echo -e "${COLOR_RED}错误: 未识别分组 '$base_arg'${COLOR_NC}"
        usage_exit
    fi

    # 添加任务
    if [ -n "$model_filter" ]; then
        TASKS+=("${group_idx}:${model_filter}")
    else
        TASKS+=("${group_idx}")
    fi
}

if [ $# -eq 0 ]; then
    # 没指定分组 → 全部
    for ((i=0; i<TOTAL_DEF_GROUPS; i++)); do
        TASKS+=("$i")
    done
else
    for arg in "$@"; do
        parse_arg "$arg"
    done
fi

# 去重
IFS=$'\n' TASKS=($(printf '%s\n' "${TASKS[@]}" | sort -u))
unset IFS

TOTAL_TASKS=${#TASKS[@]}

# --- 日志目录 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEQ_LOG_DIR="${SCRIPT_DIR}/seq_logs_${TIMESTAMP}"
mkdir -p "$SEQ_LOG_DIR"

# --- 结果记录 ---
declare -a GROUP_STATUS
PASS_GROUPS=0
FAIL_GROUPS=0
START_ALL=$(date +%s)

echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"
echo -e "${COLOR_YELLOW}MX-01 分步测评 | 平台: ${PLATFORM_ENV} | 模式: ${MODE}${COLOR_NC}"
echo -e "${COLOR_YELLOW}共 ${TOTAL_TASKS} 个任务 | 日志: ${SEQ_LOG_DIR}${COLOR_NC}"
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"

# 列出将要执行的任务
echo -e "${COLOR_CYAN}计划执行:${COLOR_NC}"
run_idx=1
for task in "${TASKS[@]}"; do
    t_idx="${task%%:*}"
    t_filter=""
    [[ "$task" == *:* ]] && t_filter="${task#*:}"
    IFS='|' read -r short args cn env <<< "${TEST_GROUPS[$t_idx]}"
    if [ -n "$t_filter" ]; then
        echo -e "  ${run_idx}. ${cn} → 仅 ${t_filter}"
    else
        echo -e "  ${run_idx}. ${cn} (${args})"
    fi
    run_idx=$((run_idx + 1))
done
echo ""

# --- 逐任务执行 ---
run_idx=1
for task in "${TASKS[@]}"; do
    t_idx="${task%%:*}"
    t_filter=""
    [[ "$task" == *:* ]] && t_filter="${task#*:}"

    IFS='|' read -r group_name group_args group_cn group_env <<< "${TEST_GROUPS[$t_idx]}"

    # 日志文件名：有模型过滤时加上模型名
    if [ -n "$t_filter" ]; then
        LOG_FILE="${SEQ_LOG_DIR}/${group_name}_${t_filter}.log"
        REPORT_FILE="${SEQ_LOG_DIR}/${group_name}_${t_filter}_report.txt"
        task_desc="${group_cn}/${t_filter}"
    else
        LOG_FILE="${SEQ_LOG_DIR}/${group_name}.log"
        REPORT_FILE="${SEQ_LOG_DIR}/${group_name}_report.txt"
        task_desc="${group_cn}"
    fi

    echo ""
    echo -e "${COLOR_BLUE}=========================================================${COLOR_NC}"
    echo -e "${COLOR_BLUE}  ▶ 第 ${run_idx}/${TOTAL_TASKS} 个任务: ${task_desc}${COLOR_NC}"
    echo -e "${COLOR_BLUE}    域: ${group_args}${COLOR_NC}"
    [ -n "$t_filter" ] && echo -e "${COLOR_BLUE}    模型: ${t_filter}${COLOR_NC}"
    [ -n "$group_env" ] && echo -e "${COLOR_BLUE}    环境变量: ${group_env}${COLOR_NC}"
    echo -e "${COLOR_BLUE}    日志: ${LOG_FILE}${COLOR_NC}"
    echo -e "${COLOR_BLUE}=========================================================${COLOR_NC}"
    run_idx=$((run_idx + 1))

    group_start=$(date +%s)

    # 执行 run_all.sh（支持通过 group_env 传递额外环境变量）
    # 用 eval export 正确处理带引号的环境变量值
    if [ -n "$group_env" ]; then
        eval "export ${group_env}"
    fi
    if [ -n "$t_filter" ]; then
        FILTER_MODELS="$t_filter" bash run_all.sh "$MODE" $group_args 2>&1 | tee "$LOG_FILE"
        rc=${PIPESTATUS[0]}
    else
        bash run_all.sh "$MODE" $group_args 2>&1 | tee "$LOG_FILE"
        rc=${PIPESTATUS[0]}
    fi

    group_end=$(date +%s)
    group_duration=$((group_end - group_start))

    if [ $rc -eq 0 ]; then
        GROUP_STATUS+=("${task_desc}: PASS (${group_duration}s)")
        PASS_GROUPS=$((PASS_GROUPS + 1))
        echo -e "\n${COLOR_GREEN}✅ ${task_desc} 完成 (耗时 ${group_duration}s)${COLOR_NC}"
    else
        GROUP_STATUS+=("${task_desc}: FAIL (${group_duration}s, rc=$rc)")
        FAIL_GROUPS=$((FAIL_GROUPS + 1))
        echo -e "\n${COLOR_RED}❌ ${task_desc} 失败 (耗时 ${group_duration}s, rc=$rc)${COLOR_NC}"
        if [ -n "$t_filter" ]; then
            echo -e "${COLOR_RED}   复测命令: FILTER_MODELS=$t_filter bash run_all.sh $MODE $group_args${COLOR_NC}"
        else
            echo -e "${COLOR_RED}   复测命令: bash run_all.sh $MODE $group_args${COLOR_NC}"
        fi
    fi

    # 从日志中提取简要报告
    {
        echo "========================================="
        echo "  ${task_desc} 测试报告"
        echo "  模式: ${MODE} | 耗时: ${group_duration}s"
        echo "========================================="
        sed -n '/^MX-01 模型能力测评报告$/,/^详细日志目录:/p' "$LOG_FILE" 2>/dev/null || true
        echo ""
    } > "$REPORT_FILE"

    echo -e "${COLOR_CYAN}  📋 报告已保存: ${REPORT_FILE}${COLOR_NC}"
    echo -e "${COLOR_CYAN}  📋 完整日志: ${LOG_FILE}${COLOR_NC}"
done

# --- 汇总 ---
END_ALL=$(date +%s)
TOTAL_DURATION=$((END_ALL - START_ALL))

echo ""
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"
echo -e "${COLOR_YELLOW}                分步测评汇总报告${COLOR_NC}"
echo -e "${COLOR_YELLOW}  平台: ${PLATFORM_ENV} | 模式: ${MODE} | 总耗时: ${TOTAL_DURATION}s${COLOR_NC}"
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"
echo ""

idx=1
for status in "${GROUP_STATUS[@]}"; do
    if echo "$status" | grep -q "PASS"; then
        echo -e "  ${COLOR_GREEN}$idx. ✅ $status${COLOR_NC}"
    else
        echo -e "  ${COLOR_RED}$idx. ❌ $status${COLOR_NC}"
    fi
    idx=$((idx + 1))
done

echo ""
echo -e "  通过: ${COLOR_GREEN}${PASS_GROUPS}${COLOR_NC}/${TOTAL_TASKS}  失败: ${COLOR_RED}${FAIL_GROUPS}${COLOR_NC}/${TOTAL_TASKS}"
echo ""
echo -e "  日志目录: ${SEQ_LOG_DIR}/"
echo ""

# 生成汇总报告文件
SUMMARY_FILE="${SEQ_LOG_DIR}/summary.txt"
{
    echo "MX-01 分步测评汇总"
    echo "平台: ${PLATFORM_ENV} | 模式: ${MODE} | 日期: $(date +%Y-%m-%d)"
    echo "总耗时: ${TOTAL_DURATION}s"
    echo ""
    idx=1
    for status in "${GROUP_STATUS[@]}"; do
        echo "  $idx. $status"
        idx=$((idx + 1))
    done
    echo ""
    echo "通过: ${PASS_GROUPS}/${TOTAL_TASKS}  失败: ${FAIL_GROUPS}/${TOTAL_TASKS}"
    echo ""
    echo "各组报告:"
    for task in "${TASKS[@]}"; do
        t_idx="${task%%:*}"
        t_filter=""
        [[ "$task" == *:* ]] && t_filter="${task#*:}"
        IFS='|' read -r group_name group_args group_cn group_env <<< "${TEST_GROUPS[$t_idx]}"
        if [ -n "$t_filter" ]; then
            echo "  ${group_cn}/${t_filter}: ${SEQ_LOG_DIR}/${group_name}_${t_filter}_report.txt"
        else
            echo "  ${group_cn}: ${SEQ_LOG_DIR}/${group_name}_report.txt"
        fi
    done
} > "$SUMMARY_FILE"

echo -e "  ${COLOR_GREEN}汇总报告: ${SUMMARY_FILE}${COLOR_NC}"
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"

if [ $FAIL_GROUPS -gt 0 ]; then
    echo -e "\n${COLOR_YELLOW}复测失败任务:${COLOR_NC}"
    for task in "${TASKS[@]}"; do
        t_idx="${task%%:*}"
        t_filter=""
        [[ "$task" == *:* ]] && t_filter="${task#*:}"
        IFS='|' read -r group_name group_args group_cn group_env <<< "${TEST_GROUPS[$t_idx]}"
        desc="${group_cn}"
        [ -n "$t_filter" ] && desc="${group_cn}/${t_filter}"
        for status in "${GROUP_STATUS[@]}"; do
            if echo "$status" | grep -q "^${desc}: FAIL"; then
                local retry_env=""
                [ -n "$group_env" ] && retry_env="$group_env "
                if [ -n "$t_filter" ]; then
                    echo -e "  ${COLOR_CYAN}${retry_env}FILTER_MODELS=$t_filter bash run_all.sh $MODE $group_args${COLOR_NC}"
                else
                    echo -e "  ${COLOR_CYAN}${retry_env}bash run_all.sh $MODE $group_args${COLOR_NC}"
                fi
                break
            fi
        done
    done
    echo ""
fi

exit $([ $FAIL_GROUPS -gt 0 ] && echo 1 || echo 0)
