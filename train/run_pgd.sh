#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

for tn in "ag_news"
do
for s in 10 2
do
python run_glue_pgd.py \
    --dataset_name $tn \
    --epochs 10 \
    --adv-steps $s \
    --bsz 16 \
    --num-labels 4 \
    --valid test \
    --ckpt-dir ../saved_models_pgd
done
done