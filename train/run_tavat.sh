#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

for tn in "mrpc"
do
for s in 5 10 2
do
python runglue_tavat.py \
    --task_name $tn \
    --epochs 10 \
    --adv_steps $s \
    --num_labels 2 \
    --bsz 16 \
    --ckpt_dir ../saved_models_tavat
done
done
