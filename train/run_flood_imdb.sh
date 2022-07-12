#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

valid="test"

python ./run_flood.py \
    --dataset_name imdb \
    --valid $valid \
    --bsz 16 \
    --eval_size 32 \
    --epochs 10 \
    --num_labels 2 \
    --b 0.02 \
    --ckpt_dir ../saved_models/
