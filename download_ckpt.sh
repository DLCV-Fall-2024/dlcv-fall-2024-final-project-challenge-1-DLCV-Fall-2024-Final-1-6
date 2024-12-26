#!bin/bash

mkdir -p ckpt
cd ckpt
gdown 11GW_2auEw_7W9cALvAPnHRacgxvV7Ck- -O stage1_ckpt.zip
gdown 118n5zB-vU17-FwLbDdljwIeKpU9JMj0X -O stage2_ckpt.zip
gdown 1BSLZD9yk9YVjDpdPd2I7ExjFG2Cc06gn -O yolo11x.pt
unzip stage1_ckpt.zip
mv checkpoint-600 stage1_ckpt
unzip stage2_ckpt.zip
mv checkpoint-3000 stage2_ckpt
rm stage1_ckpt.zip
rm stage2_ckpt.zip
cd ..
