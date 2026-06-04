#!/bin/bash

set -e

export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1}

echo "Evaluate ESPCN START"

python test_image.py --upscale_factor 2 --model_name epoch_3_100.pt

echo "Evaluate ESPCN FINISHED"
