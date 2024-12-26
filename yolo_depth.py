import os
import json
import sys

import numpy as np
import torch
import cv2
from torch.utils.data import IterableDataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from transformers import pipeline
from ultralytics import YOLO
from datasets import load_dataset


# Constants
YOLO_MODEL_PATH = "weights/yolov8x-worldv2.pt"
DEPTH_MODEL_PATH = "depth-anything/Depth-Anything-V2-large-hf"
CLASSES = [
    "car", "truck", "bus", "motorcycle", "bicycle", "tricycle", "van", "suv", "trailer", 
    "construction vehicle", "moped", "recreational vehicle", "pedestrian", "cyclist", 
    "wheelchair", "stroller", "traffic light", "traffic sign", "traffic cone", 
    "traffic island", "traffic box", "barrier", "bollard", "warning sign", "debris", 
    "machinery", "dustbin", "concrete block", "cart", "chair", "basket", "suitcase", 
    "dog", "phone booth"
]
OUTPUT_FILE = "results/test_detection.json"
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Create results directory
os.makedirs("results", exist_ok=True)

def load_our_dataset(split):
    """Load the dataset for the given split."""
    return load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)

# Load datasets
training_dataset: IterableDataset = load_our_dataset("train")
val_dataset: IterableDataset = load_our_dataset("val")
test_dataset: IterableDataset = load_our_dataset("test")

def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    ids = [item["id"] for item in batch]
    images = [item["image"] for item in batch]
    conversations = [item["conversations"] for item in batch]
    
    return {
        "ids": ids,
        "images": images,
        "conversations": conversations
    }

def get_depth_category(depth_value):
    """Get depth category based on depth value."""
    thresholds = {
        1.0: "immediate",
        0.6: "short range",
        0.4: "mid range",
        0.15: "long range"
    }
    for threshold, category in sorted(thresholds.items()):
        if depth_value <= threshold:
            return category
def get_position(bbox, image_width):
    """
    Determine the horizontal position of an object based on its bounding box center
    """
    # Calculate center x-coordinate of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    
    # Define the boundaries for three equal sections
    third_width = image_width / 3
    
    if center_x < third_width:
        return "left"
    elif center_x < 2 * third_width:
        return "middle"
    else:
        return "right"

# Create DataLoaders
train_loader = DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# Initialize model and pipeline
data = dict()
detection_model = YOLO(YOLO_MODEL_PATH)
depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL_PATH, device=DEVICE)

# Set custom classes if defined
if CLASSES:
    detection_model.set_classes(CLASSES)

# Execute prediction for specified categories on an image
for batch_idx, batch in enumerate(tqdm(test_loader)):
    yolo_results = detection_model.predict(batch["images"], conf=0.25)
    depth_results = depth_pipe(batch["images"])
    depth_map = np.array(depth_results[0]["predicted_depth"])
        
    # Resize depth map to match image size
    w, h = batch["images"][0].size
    depth_map_tensor = torch.tensor(depth_map)
    depth_map = F.interpolate(depth_map_tensor[None, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_map = depth_map.cpu().numpy()
    depth_max=depth_map.max()
    depth_min=depth_map.min()
    

    for index, (yolo_result, depth_result) in enumerate(zip(yolo_results, depth_results)):
        image_data = []
        for box in yolo_result.boxes:
            score = box.conf.item()
            label = box.cls.item()
            xyxy = box.xyxy[0].tolist()
            
            # pass to depth detection model
            roi_depth = depth_map[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            avg_depth = float(np.mean(roi_depth))
            avg_depth=(avg_depth-depth_min)/(depth_max-depth_min)
            position = get_position(xyxy, w)
            

            image_data.append({
                "bbox_n": box.xyxyn[0].tolist(),
                "bbox": xyxy,
                "confidence": float(score),
                "depth_value": avg_depth,
                "depth_category": get_depth_category(avg_depth),
                "category_name": detection_model.names.get(box.cls.item()),
                "position":position
            })

        data[batch["ids"][index]] = image_data

        yolo_result.save_txt(f"results/{batch_idx * TEST_BATCH_SIZE + index}.txt")
        yolo_result.save(f"results/{batch_idx * TEST_BATCH_SIZE + index}.jpg")

# Save results to JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)
    print(f"Saved to {OUTPUT_FILE}")