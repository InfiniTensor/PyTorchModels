# ==============================================================================
# 统一环境配置：选择目标平台 + 数据集根目录 + 可见设备
#
# 用法：先按本机情况设置 PLATFORM_ENV / DATASET_ROOT，再 source 本文件
#       source env.sh
# 也可在命令行直接覆盖（优先级高于本文件默认值）：
#       PLATFORM_ENV=ASCEND_NPU DATASET_ROOT=/data/Dataset CUDA_VISIBLE_DEVICES=0,1 source env.sh
#
# 支持的平台 (PLATFORM_ENV)，与 usercustomize.py 的导入钩子一一对应：
#   NVIDIA_GPU      NVIDIA（基线，纯 CUDA）
#   ASCEND_NPU      华为昇腾（import torch_npu；需在各机器 source 昇腾 set_env.sh）
#   CAMBRICON_MLU   寒武纪（import torch_mlu；需 sitecustomize.py 设 ENABLE_USER_SITE=True）
#   HYGON_DCU       海光 / 曙光 DCU（与 SUGON_DCU 等价，纯 CUDA/ROCm 兼容）
#   ILLUVATAR_GPU   天数智芯（纯 CUDA 兼容）
#   METAX_GPU       沐曦（默认纯 CUDA 兼容；若需 musa 见 usercustomize.py 注释）
#   MOORE_GPU       摩尔线程（import torch_musa）
# ==============================================================================

# ---- 1) 目标平台 ----
export PLATFORM_ENV="${PLATFORM_ENV:-NVIDIA_GPU}"

# ---- 2) 数据集物理根目录 ----
# 所有 run_*.sh / .py 中的数据路径统一通过 $DATASET_ROOT 引用，换机器只改这里。
# 兼容旧变量 DATA_DIR：若 DATASET_ROOT 未设而 DATA_DIR 已设，则沿用 DATA_DIR。
if [ -z "${DATASET_ROOT:-}" ] && [ -n "${DATA_DIR:-}" ]; then
    export DATASET_ROOT="${DATA_DIR}"
fi
export DATASET_ROOT="${DATASET_ROOT:-/data1/shared/Dataset}"

# ---- 3) 可见设备（统一以 CUDA_VISIBLE_DEVICES 为真源）----
# 若命令行已指定（如 CUDA_VISIBLE_DEVICES=4,5 ./run_sequential.sh），则不覆盖
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="0,1"
fi

# ---- 4) Torch hub 缓存目录（避免根分区空间不足）----
# 每台机器可按需覆盖，默认用用户家目录下的缓存
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# ---- 5) 按平台把可见设备映射到厂商原生变量 ----
# usercustomize.py 统一读 CUDA_VISIBLE_DEVICES；这里额外同步到厂商变量，
# 供厂商工具链或非 Python 侧（如 shell 控制的进程）使用。
case "$PLATFORM_ENV" in
    ASCEND_NPU)
        # 前置：请在各昇腾机器上按需取消注释并改为本机 toolkit 路径
        # source /usr/local/Ascend/ascend-toolkit/set_env.sh
        [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export ASCEND_RT_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
        ;;
    CAMBRICON_MLU)
        [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export MLU_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
        ;;
    MOORE_GPU)
        [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export MUSA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
        ;;
    NVIDIA_GPU|HYGON_DCU|SUGON_DCU|ILLUVATAR_GPU|METAX_GPU)
        # 纯 CUDA 兼容，直接使用 CUDA_VISIBLE_DEVICES，无需映射
        ;;
    *)
        echo "env.sh: 警告：未知 PLATFORM_ENV='${PLATFORM_ENV}'，按 NVIDIA/CUDA 兼容处理" >&2
        ;;
esac

# ---- 6) 昇腾反向映射：仅设了 ASCEND_RT_VISIBLE_DEVICES 时回填 CUDA_VISIBLE_DEVICES ----
if [ "$PLATFORM_ENV" = "ASCEND_NPU" ] && [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
fi
