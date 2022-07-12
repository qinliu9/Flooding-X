#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
log_file="normal_sst2_0928.log"

python run_glue.py \
    --dataset_name glue \
    --task_name sst2 \
    --model_name roberta-base \
    --bias_correction True \
    --epochs 10 \
    --ckpt_dir ../saved_models_roberta