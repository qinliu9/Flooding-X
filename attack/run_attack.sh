#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

model_name='attention_freelb_bert-base-uncased_sst2_adv2_lam0.1_epochs10'
log_file='../results/'$model_name'.log'
log_dir='../results/'$model_name'.csv'


for epoch in 9 8

do
  #echo "freelb_stifiness_adv_${adv} epoch_${total} mag_${mag}: epoch ${epoch}" | tee -a $log_file
  python ./attack_finetune.py \
  --task_name sst2 \
  --results_file $log_dir \
  --num_examples 1000 \
  --model_name_or_path ../saved_models/$model_name/'epoch'$epoch >> ${log_file}
done
