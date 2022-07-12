#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
log_file="freelb_sst2_1002.log"

python run_glue_freelb.py \
    --dataset_name glue \
    --task_name sst2 \
    --epochs 10 \
    --adv_steps 5 \
    --ckpt_dir ../saved_models >> ${log_file}