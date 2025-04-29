export CUDA_VISIBLE_DEVICES=4,5
python maskrcnn.py --mode train \
  --train-epochs 2 \
  --train-batch-size 8 \
  --device cuda \
  --dataset-root ../data/coco
