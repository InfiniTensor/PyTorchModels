model=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 单机四卡分布式训练
python main.py \
    -a $model \
    --dist-backend 'nccl' \
    --dist-url "tcp://localhost:8828" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --batch-size 64 \
    --profile \
    /data1/shared/Dataset/imagenet2012
