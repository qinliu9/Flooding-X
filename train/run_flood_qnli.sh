#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1


python ./run_flood.py \
    --task_name qnli \
    --bsz 16 \
    --eval_size 32 \
    --epochs 10 \
    --b 0.036 \
    --ckpt_dir ../saved_models/
