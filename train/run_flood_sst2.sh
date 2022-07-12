#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

for flooding in 0.1 0.2
  do
    python ./run_flood.py \
        --dataset_name glue \
        --task_name sst2 \
        --epochs 10 \
        --b $flooding \
        --bsz 32 \
        --ckpt_dir ../saved_models_flooding_sst2/
done
