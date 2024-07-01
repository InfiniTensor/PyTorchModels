model=$1
export CUDA_VISIBLE_DEVICES=0

# 单机单卡推理
python main.py \
    -a $model \
    --dist-backend 'nccl' \
    --dist-url "tcp://localhost:8828" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --batch-size 64 \
    --weights alexnet-owt-7be5be79.pth \
    --evaluate \
    /data1/shared/Dataset/imagenet2012
