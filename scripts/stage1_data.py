import gc
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import json
import os
import argparse
from datasets import load_dataset
from ultralytics import YOLO
from stage1_utils import generate_3d_spatial_graph
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Process images with YOLO and depth estimation')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, 
                        default='ntudlcv/dlcv_2024_final1',
                        help='Name of the dataset to load')
    parser.add_argument('--split', type=str,
                        choices=['train', 'val', 'test'],
                        default='test',
                        help='Dataset split to process')
    
    # Model arguments
    parser.add_argument('--yolo_checkpoint', type=str,
                        default='checkpoint/yolo11x.pt',
                        help='Path to YOLO model checkpoint')
    parser.add_argument('--depth_model', type=str,
                        default='depth-anything/Depth-Anything-V2-Small-hf',
                        help='Name/path of depth estimation model')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                        default='./stage1_test_general',
                        help='Directory to save processed data')
    parser.add_argument('--save_images', action='store_true',
                        help='Whether to save original images')
    parser.add_argument('--skip_duplicates', action='store_true',   
                        help='Whether to skip duplicate images')
    
    return parser.parse_args()


def setup_yolo(checkpoint_path):
    """Initialize YOLO model"""
    return YOLO(checkpoint_path)


def setup_depth(model_name):
    return pipeline(task="depth-estimation", model=model_name, device=0)


def YOLO_inference(image, model, pipe):
    """Run YOLO inference on an image"""
    with torch.no_grad():
        result = generate_3d_spatial_graph(image, yolo_model=model, depth_pipe=pipe)
    return result


def process_dataset(dataset, yolo_model, depth_model, output_dir, save_images=False, test=False, skip=True):
    """Process dataset with YOLO model"""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    processed_data = []

    for idx, item in enumerate(tqdm(dataset)):
        if skip and "general" not in item["id"]:
            continue
        
        # Get image and save it
        image = item["image"]
        image_filename = f"images/image_{idx}.jpg"
        
        # Save original image if requested
        if save_images:
            image.save(os.path.join(output_dir, image_filename))
        
        # Process with YOLO
        yolo_result = YOLO_inference(image, yolo_model, depth_model)
        
        question_prompt = "Now please describe all the objects, and explain why it affects the ego car's behavior."
        new_prompt = "<image>\nThere is an image of traffic captured from the perspective of the ego car. "+\
              "The rough detection of some of the objects and their relative locations in the image are listed below: " + \
              yolo_result + " " + question_prompt
        
        # Create data entry
        if test:
            data_entry = {
                "id": idx,
                "realid": item["id"],
                "image": image_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": new_prompt
                    }
                ]
            }
        else:
            data_entry = {
                "id": idx,
                "realid": item["id"],
                "image": image_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": new_prompt
                    },
                    {
                        "from": "gpt",
                        "value": item["conversations"][1]["value"]
                    }
                ]
            }
        processed_data.append(data_entry)

    # Save processed data to JSON
    with open(os.path.join(output_dir, "processed_data.json"), 'w') as f:
        json.dump(processed_data, f, indent=4)


def main():
    args = parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, num_proc=4)
    
    # Initialize models
    yolo_model = setup_yolo(args.yolo_checkpoint)
    depth_model = setup_depth(args.depth_model)
    
    # Process data
    is_test = args.split == 'test'
    process_dataset(
        dataset[args.split], 
        yolo_model, 
        depth_model, 
        args.output_dir,
        save_images=args.save_images,
        test=is_test,
        skip=args.skip_duplicates
    )


if __name__ == "__main__":
    main()
