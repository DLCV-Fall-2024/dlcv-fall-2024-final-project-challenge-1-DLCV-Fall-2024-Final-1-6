# DLCV Final Project

# How to run your code?

## Setup

1. Clone the repository

``` bash
git clone https://github.com/DLCV-Fall-2024/DLCV-Fall-2024-Final-1-cvpr2025.git
cd DLCV-Fall-2024-Final-1-cvpr2025
```

2. Create a new conda environment

``` bash
conda create -n final python=3.10.16 --y
conda activate final

# Install the required packages
cd LLaVA
pip install -e .
pip insatll -e ".[train]"
pip install flash-attn
cd ..
```

## Data generation

1. Download yolo weights. Please refer to the README.md in the weights folder.

2. Create another environment for data generation using python 3.11

``` bash
conda create -n gen_data python=3.11 --y
conda activate gen_data

pip install -r generate_pretrain_data/requirements.txt
```

3. Generate the annotation file for training and testing

``` bash

bash ./gen_annotation.sh <Path to processed data folder> <split>

# Example
# bash ./gen_annotation.sh ./training train
# bash ./gen_annotation.sh ./testing test
```

- first argument - path to the processed data folder
- second argument - should be either `train`, `val`, or `test`.


# Training

``` bash
# At the root of the repository (DLCV-Fall-2024-Final-1-cvpr2025)

bash ./train.sh <training_annotation_path> <validation_annotation_path> <model_checkpoint_path> <pretrain_bbox_encoder_path>

# Example
# bash ./train.sh ./training/train.json ./validation/val.json ./model_checkpoint ./pretrain_bbox_encoder
```

- first argument - path to the training annotation file
- second argument - path to the validation annotation file (currently deprecrated for llava)
- third argument - path to the model checkpoint folder (will be created if not exists)
- fourth argument - path to the pretrain bbox encoder folder (if pretraining pretrained bbox encoder model exists)

This will generate the model checkpoints in the `model_checkpoint` folder with 6000 steps currently.

# Inferencing

``` bash    
# At the root of the repository (DLCV-Fall-2024-Final-1-cvpr2025)
bash ./inference.sh <model_checkpoint_path> <testing_annotation> <test_images> <output_json>

# Example
# bash ./inference.sh ./model_checkpoint ./testing/test.json ./testing/test_images ./submission.json
```

- first argument - path to the model checkpoint folder
- second argument - path to the testing annotation file
- third argument - path to the testing images folder
- fourth argument - path to the output json file

This will generate the output json file in the output folder in the fourth argument.

# Pretraining

1. Download the dataset from https://www.nuscenes.org/nuimages (our main dataset used for pretraining) 

2. move all the images to a folder named `nuImages`.

``` bash
mv <donwloaded_folder> ./nuImages

# If there are multiple folders, move all the images to the same folder
```

3. generate pretrained annotation data
``` bash
# At the root of the repository (DLCV-Fall-2024-Final-1-cvpr2025)
bash ./gen_pretrain_annotation.sh <path_to_nuImages> <output_folder>

# Example
# bash ./gen_pretrain_annotation.sh ./nuImages ./pretrain_data
```

- first argument - path to the nuImages folder
- second argument - path to the output folder

This will create a file named `pretrain_data.json` in the output folder.

4. Pretrain the model

``` bash
# At the root of the repository (DLCV-Fall-2024-Final-1-cvpr2025)
bash ./pretrain.sh <training_annotation_path>  <output_checkpoint_path>

# Example
# bash ./pretrain.sh ./pretrain_data/pretrain_data.json ./pretrain_checkpoint
```

- first argument - path to the pretrain annotation file
- second argument - path to the output checkpoint folder

This will generate the pretrained model checkpoints in the `pretrain_checkpoint` folder with 2000 steps currently.