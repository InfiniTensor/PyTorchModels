export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python pmnist_test.py
