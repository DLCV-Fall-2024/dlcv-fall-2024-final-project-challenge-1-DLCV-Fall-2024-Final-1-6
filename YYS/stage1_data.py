import gc
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import json
import os
from datasets import load_dataset
from ultralytics import YOLO


def setup_yolo():
    """Initialize YOLO model"""
    return YOLO("./yolo11x_1216.pt")

def YOLO_inference(image, model):
    """Run YOLO inference on an image"""
    with torch.no_grad():
        results = model(image, stream=False)
        def rounding(tensor):
            mid = tensor.tolist()
            result = [f"{round(mid[0], 2):.2f}"[1:], f"{round(mid[1], 2):.2f}"[1:], 
                     f"{round(mid[2], 2):.2f}"[1:], f"{round(mid[3], 2):.2f}"[1:]]
            return f"{result[0]}, {result[1]}, {result[2]}, {result[3]}"
        return [[f'{result.names[result.boxes.cls[i].item()]} in {rounding(result.boxes.xyxyn[i])}' 
                for i in range(result.boxes.cls.shape[0])] for result in results]

def process_dataset(dataset, yolo_model, output_dir):
    """Process dataset with YOLO model"""
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []

    for idx, item in enumerate(tqdm(dataset)):
        # Skip if id doesn't contain "general"
        if "general" not in item["id"]:
            continue
        
        # Get image and save it
        image = item["image"]
        image_filename = f"image_{idx}.jpg"
        
        
        # Save original image
        image.save(os.path.join(output_dir, image_filename))
        
        # Process with YOLO
        yolo_results = YOLO_inference(image, yolo_model)
        
        # Create data entry
        data_entry = {
            "id": idx,
            "image": image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": f"This is a traffic scene image. I want you to analyze the image and tell me what objects you can see in the image. YOLO object detection shows: {yolo_results}. Please provide a detailed description of the scene, including the locations and relationships of the objects."
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
        json.dump(processed_data, f, indent=2)

def main():
    
    # Load dataset
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", num_proc=4)
    
    # Initialize YOLO model
    yolo_model = setup_yolo()
    
    # Process training data
    output_dir = "./stage1_train_general"
    process_dataset(dataset["train"], yolo_model, output_dir)

if __name__ == "__main__":
    main()
