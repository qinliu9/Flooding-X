#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

for tn in "sst2"
do
for s in 2 5
do
python runglue_infobert.py \
    --task_name $tn \
    --epochs 10 \
    --num_labels 2 \
    --adv_steps $s \
    --bsz 32 \
    --ckpt_dir ../saved_models_infobert
done
done