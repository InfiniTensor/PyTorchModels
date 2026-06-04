#!/bin/bash

set -e

export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1}

echo "Evaluate TCN START"

# TCN uses pmnist_test.py which includes train+test cycles.
# For eval-only, run with 1 epoch and measure inference throughput.
python pmnist_test.py --epochs 1

echo "Evaluate TCN FINISHED"
