#!/bin/bash

model_path="/home/twszmxa453/CVPR2025/checkpoints/llava-v1.5-7b-finetune-with-pretrain-lora-1223"
image_folder="/home/twszmxa453/CVPR2025/dlcv_test_data/test_images"
output_json=$2
annotation_file=$1

base_model="liuhaotian/llava-v1.5-7b"

python LLaVA/llava/eval/gen_car_output.py \
    --model-base $base_model \
    --model-path $model_path \
    --image-folder $image_folder \
    --output_file $output_json \
    --annotation_file $annotation_file