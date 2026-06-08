#!/bin/bash

# ==============================================================================
#                     MX-01 分步测评脚本
#           按类别依次测试，每类独立日志，方便截图和复测
#
#   用法:
#     ./run_sequential.sh                              # 全部分步执行
#     ./run_sequential.sh train                        # 仅训练（分步）
#     ./run_sequential.sh eval                         # 仅推理（分步）
#     ./run_sequential.sh all 1 3                      # 跑第1、3组（训练+推理）
#     ./run_sequential.sh train 4-6                    # 跑第4到6组（仅训练）
#     ./run_sequential.sh eval Detection Speech        # 按名称跑指定域（仅推理）
#
#   指定组的方式（可混用）:
#     数字编号:   1 3 5        第1、3、5组
#     范围:       2-4          第2到4组
#     域名:       Detection    按名称匹配
#
#   测试分组:
#     1. single     单模型合集 (GAN + NLP + RL + Recommendation + SR)   共 5 模型
#     2. ts         TimeSeriesPrediction                                 共 2 模型
#     3. speech     Speech                                               共 2 模型
#     4. det        Detection                                            共 3 模型
#     5. seg        Segmentation                                         共 4 模型
#     6. ic         ImageClassification                                  共 60 模型
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
# "分组简称|run_all.sh参数|中文名"
declare -a TEST_GROUPS=(
    "single|GAN NLP RL Recommendation SR|单模型合集(GAN+NLP+RL+Rec+SR)"
    "ts|TimeSeriesPrediction|时序预测"
    "speech|Speech|语音识别"
    "det|Detection|目标检测"
    "seg|Segmentation|语义分割"
    "ic|ImageClassification|图像分类"
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

# --- 选择要跑的分组 ---
declare -a SELECTED_INDICES  # 存选中分组的下标

usage_exit() {
    echo -e "${COLOR_RED}用法: $0 [all|train|eval] [分组...]${COLOR_NC}"
    echo ""
    echo "  指定分组的方式（可混用）:"
    echo "    数字编号    1 3 5       第1、3、5组"
    echo "    范围        2-4         第2到4组"
    echo "    域名/简称   Detection   按名称匹配"
    echo ""
    echo "  可用分组:"
    local i=1
    for g in "${TEST_GROUPS[@]}"; do
        IFS='|' read -r short args cn <<< "$g"
        printf "    %-3s %-22s %s\n" "$i." "$short" "$cn"
        i=$((i + 1))
    done
    exit 1
}

if [ $# -eq 0 ]; then
    # 没指定分组 → 全部
    for ((i=0; i<TOTAL_DEF_GROUPS; i++)); do
        SELECTED_INDICES+=($i)
    done
else
    for arg in "$@"; do
        # 1) 范围: 2-4
        if [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local_start=${BASH_REMATCH[1]}
            local_end=${BASH_REMATCH[2]}
            if [ "$local_start" -lt 1 ] || [ "$local_end" -gt "$TOTAL_DEF_GROUPS" ] || [ "$local_start" -gt "$local_end" ]; then
                echo -e "${COLOR_RED}错误: 无效范围 '$arg' (有效: 1-${TOTAL_DEF_GROUPS})${COLOR_NC}"
                usage_exit
            fi
            for ((i=local_start; i<=local_end; i++)); do
                SELECTED_INDICES+=($((i - 1)))
            done
        # 2) 数字编号: 1 3 5
        elif [[ "$arg" =~ ^[0-9]+$ ]]; then
            if [ "$arg" -lt 1 ] || [ "$arg" -gt "$TOTAL_DEF_GROUPS" ]; then
                echo -e "${COLOR_RED}错误: 无效编号 '$arg' (有效: 1-${TOTAL_DEF_GROUPS})${COLOR_NC}"
                usage_exit
            fi
            SELECTED_INDICES+=($((arg - 1)))
        # 3) 名称/简称匹配
        else
            local found=0
            for ((i=0; i<TOTAL_DEF_GROUPS; i++)); do
                IFS='|' read -r short args cn <<< "${TEST_GROUPS[$i]}"
                # 匹配简称、中文名、或 run_all.sh 参数中的任意域名
                if [ "$arg" = "$short" ] || echo "$args" | grep -qw "$arg"; then
                    SELECTED_INDICES+=($i)
                    found=1
                    break
                fi
            done
            if [ $found -eq 0 ]; then
                echo -e "${COLOR_RED}错误: 未识别分组 '$arg'${COLOR_NC}"
                usage_exit
            fi
        fi
    done
fi

# 去重并排序
IFS=$'\n' SELECTED_INDICES=($(echo "${SELECTED_INDICES[*]}" | tr ' ' '\n' | sort -nu | uniq))
unset IFS

# --- 日志目录 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEQ_LOG_DIR="${SCRIPT_DIR}/seq_logs_${TIMESTAMP}"
mkdir -p "$SEQ_LOG_DIR"

# --- 结果记录 ---
declare -a GROUP_STATUS
TOTAL_GROUPS=${#SELECTED_INDICES[@]}
PASS_GROUPS=0
FAIL_GROUPS=0
START_ALL=$(date +%s)

echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"
echo -e "${COLOR_YELLOW}MX-01 分步测评 | 平台: ${PLATFORM_ENV} | 模式: ${MODE}${COLOR_NC}"
echo -e "${COLOR_YELLOW}共 ${TOTAL_GROUPS} 个测试组 | 日志: ${SEQ_LOG_DIR}${COLOR_NC}"
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"

# 列出将要执行的分组
echo -e "${COLOR_CYAN}计划执行:${COLOR_NC}"
run_idx=1
for i in "${SELECTED_INDICES[@]}"; do
    IFS='|' read -r short args cn <<< "${TEST_GROUPS[$i]}"
    echo -e "  ${run_idx}. ${cn} (${args})"
    run_idx=$((run_idx + 1))
done
echo ""

# --- 逐组执行 ---
run_idx=1
for i in "${SELECTED_INDICES[@]}"; do
    IFS='|' read -r group_name group_args group_cn <<< "${TEST_GROUPS[$i]}"

    LOG_FILE="${SEQ_LOG_DIR}/${group_name}.log"
    REPORT_FILE="${SEQ_LOG_DIR}/${group_name}_report.txt"

    echo ""
    echo -e "${COLOR_BLUE}=========================================================${COLOR_NC}"
    echo -e "${COLOR_BLUE}  ▶ 第 ${run_idx}/${TOTAL_GROUPS} 组: ${group_cn}${COLOR_NC}"
    echo -e "${COLOR_BLUE}    域: ${group_args}${COLOR_NC}"
    echo -e "${COLOR_BLUE}    日志: ${LOG_FILE}${COLOR_NC}"
    echo -e "${COLOR_BLUE}=========================================================${COLOR_NC}"
    run_idx=$((run_idx + 1))

    group_start=$(date +%s)

    # 执行 run_all.sh，tee 同时输出到屏幕和日志文件
    bash run_all.sh "$MODE" $group_args 2>&1 | tee "$LOG_FILE"
    rc=${PIPESTATUS[0]}

    group_end=$(date +%s)
    group_duration=$((group_end - group_start))

    if [ $rc -eq 0 ]; then
        GROUP_STATUS+=("${group_cn}: PASS (${group_duration}s)")
        PASS_GROUPS=$((PASS_GROUPS + 1))
        echo -e "\n${COLOR_GREEN}✅ ${group_cn} 完成 (耗时 ${group_duration}s)${COLOR_NC}"
    else
        GROUP_STATUS+=("${group_cn}: FAIL (${group_duration}s, rc=$rc)")
        FAIL_GROUPS=$((FAIL_GROUPS + 1))
        echo -e "\n${COLOR_RED}❌ ${group_cn} 失败 (耗时 ${group_duration}s, rc=$rc)${COLOR_NC}"
        echo -e "${COLOR_RED}   复测命令: bash run_all.sh $MODE $group_args${COLOR_NC}"
    fi

    # 从日志中提取简要报告
    {
        echo "========================================="
        echo "  ${group_cn} 测试报告"
        echo "  模式: ${MODE} | 耗时: ${group_duration}s"
        echo "========================================="
        # 提取 run_all.sh 的控制台报告部分
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
echo -e "  通过: ${COLOR_GREEN}${PASS_GROUPS}${COLOR_NC}/${TOTAL_GROUPS}  失败: ${COLOR_RED}${FAIL_GROUPS}${COLOR_NC}/${TOTAL_GROUPS}"
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
    echo "通过: ${PASS_GROUPS}/${TOTAL_GROUPS}  失败: ${FAIL_GROUPS}/${TOTAL_GROUPS}"
    echo ""
    echo "各组报告:"
    for i in "${SELECTED_INDICES[@]}"; do
        IFS='|' read -r group_name group_args group_cn <<< "${TEST_GROUPS[$i]}"
        echo "  ${group_cn}: ${SEQ_LOG_DIR}/${group_name}_report.txt"
    done
} > "$SUMMARY_FILE"

echo -e "  ${COLOR_GREEN}汇总报告: ${SUMMARY_FILE}${COLOR_NC}"
echo -e "${COLOR_YELLOW}=========================================================${COLOR_NC}"

if [ $FAIL_GROUPS -gt 0 ]; then
    echo -e "\n${COLOR_YELLOW}复测失败分组:${COLOR_NC}"
    for i in "${SELECTED_INDICES[@]}"; do
        IFS='|' read -r group_name group_args group_cn <<< "${TEST_GROUPS[$i]}"
        for status in "${GROUP_STATUS[@]}"; do
            if echo "$status" | grep -q "^${group_cn}: FAIL"; then
                echo -e "  ${COLOR_CYAN}bash run_all.sh $MODE $group_args${COLOR_NC}"
                break
            fi
        done
    done
    echo ""
fi

exit $([ $FAIL_GROUPS -gt 0 ] && echo 1 || echo 0)
