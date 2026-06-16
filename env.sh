# Choose from ["NVIDIA_GPU", "CAMBRICON_MLU", "ASCEND_NPU",
#              "METAX_GPU", "MOORE_GPU", "SUGON_DCU", "ILLUVATAR_GPU"]
export PLATFORM_ENV="NVIDIA_GPU"

# GPU devices for benchmark (change this to control which GPUs to use)
# 如果命令行已指定（如 CUDA_VISIBLE_DEVICES=4,5 ./run_sequential.sh），则不覆盖
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1"
fi

# Torch hub cache on /data (root partition has limited space)
export TORCH_HOME=/data/shared/baoming/.cache/torch
