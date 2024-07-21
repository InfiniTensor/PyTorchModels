export CUDA_VISIBLE_DEVICES=0,1,2,3

ARCH=""
LOG_FILE=""

# Help message
usage() {
    echo "Usage: $0 -a ARCH"
    echo "  -a ARCH, --arch ARCH       Model architecture (e.g., AlexNet, ResNet18, etc.)"
    echo "  -h, --help                 Display this help and exit"
    echo "  -l LOG_FILE, --log LOG_FILE Log file"
    exit 1
}

# Parse command line arguments
while getopts "a:l:h" opt; do
    case $opt in
        a) ARCH=$OPTARG ;;
        l) LOG_FILE=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check ARCH parsing
if [ -z "$ARCH" ]; then
    echo "Error: Model architecture is required."
    usage
fi

# Check dataset
if [ -e "../data/imagenet2012" ]; then
    echo "Dataset ../data/imagenet2012 exists"
else
    # Link defaut dataset
    ln -s /data1/shared/Dataset/imagenet2012 ../data
fi

# Define log file with ARCH included
if [ -z "$LOG_FILE" ]; then
    LOG_FILE="pytorch-${ARCH}-train-gpu${CUDA_VISIBLE_DEVICES}.log"
    echo "Logs saved in defaut log file ${LOG_FILE}"
else 
    echo "Logs saved in ${LOG_FILE}"
fi

echo "Training Start: $(date +'%m/%d/%Y %T')" > ${LOG_FILE}

python main.py \
    -a $ARCH \
    --dist-backend 'nccl' \
    --dist-url "tcp://localhost:8828" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --batch-size 64 \
    ../data/imagenet2012  2>&1 | tee -a $LOG_FILE

echo "Training Finish: $(date +'%m/%d/%Y %T')" >> ${LOG_FILE}