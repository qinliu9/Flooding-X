#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
log_file="finetune_imdb_0926.log"
valid="test"

python run_glue.py \
    --dataset_name imdb \
    --valid $valid \
    --bsz 16 \
    --eval_size 32 \
    --epochs 10 \
    --num_labels 2 \
    --ckpt_dir ./saved_models/ >> ${log_file}
