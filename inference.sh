#!/bin/bash

model_path=$1
annotation_file=$2
image_folder=$3
output_json=$4

base_model="liuhaotian/llava-v1.5-7b"

python LLaVA/llava/eval/gen_car_output.py \
    --model-base $base_model \
    --model-path $model_path \
    --image-folder $image_folder \
    --output_file $output_json \
    --annotation_file $annotation_file