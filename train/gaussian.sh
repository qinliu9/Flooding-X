#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
log_file="../gaussian/new_sst2_tavat.csv"


for a in 0.3
do
python test_gaussian.py \
    --task_name sst2 \
    --num_labels 2 \
    --bsz 32 \
    --model_dir ../saved_models_tavat/TAVAT_bert-base-uncased_glue_sst2_adv10_epochs10_testlast \
    --alpha $a \
    --results_file $log_file
done