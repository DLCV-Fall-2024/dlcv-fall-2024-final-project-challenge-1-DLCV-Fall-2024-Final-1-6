#!/bin/bash

# /mnt/HDD_1/walker/dlcv_json_files/train.json
# /mnt/HDD_1/walker/dlcv_json_files/val.json
# /mnt/HDD_1/walker/dlcv_checkpoints/llava-v1.5-7b-no-pretrain-finetune-lora-1226
data_path=$1
val_data_path=$2
output_dir=$3

# can exist arguments
# /mnt/HDD_1/walker/dlcv_checkpoints/llava-v1.5-7b-pretrain-1224/checkpoint-900/bbox_encoder.bin
pretrain_bbox_encoder=${4:-""}


cd LLaVA
bash ./scripts/v1_5/finetune_task_lora.sh $data_path $val_data_path $output_dir $pretrain_bbox_encoder