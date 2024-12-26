'''
This script demonstrates how to estimate depth for an object within a bounding box using the Depth-Anything
'''
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from datasets import load_dataset
from transformers import pipeline

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-large-hf")
dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test", streaming=True)
output_dir = "b10901091/DLCV-Fall-2024-Final-1-cvpr2025/depth_output"

def estimate_depth(item, bbox, pipe, output_dir, visualize=False):
    """
    Estimate depth for an object within a bounding box and categorize it into ranges.
    
    Args:
        item: dictionary containing the image data and bbox for corresponding object from dataset
        bbox: tuple of (x1, y1, x2, y2) coordinates for the bounding box
        pipe: depth estimation pipeline
        output_dir: directory to save depth images if visualize is True
        visualize: boolean flag to save depth images for visualization
        
    Returns:
        item: dictionary with depth category added
    """
    try:
        # Get depth map for the entire image
        image = item["image"]
        width, height = image.size
        result = pipe(image)
        depth_map = np.array(result["predicted_depth"])
        depth_map_tensor = torch.tensor(depth_map)
        depth_image = F.interpolate(depth_map_tensor[None, None], (height, width), mode='bilinear', align_corners=False)[0, 0]
        depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255.0
        depth_image = depth_image.cpu().numpy().astype(np.uint8)
        
        if visualize:
            depth_image_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
            cv2.imwrite(os.path.join(output_dir, f"{item['id']}_depth.png"), depth_image_colormap)
        
        # Extract the region of interest using the bbox
        x1, y1, x2, y2 = map(int, bbox)
        roi_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(roi_depth)
        
        # Define thresholds for depth ranges
        immediate_threshold = 0.3
        short_range_threshold = 0.5
        midrange_threshold = 0.7
        
        # Categorize the depth
        if avg_depth <= immediate_threshold:
            item["depth_category"] = "immediate"
        elif avg_depth <= short_range_threshold:
            item["depth_category"] = "short range"
        elif avg_depth <= midrange_threshold:
            item["depth_category"] = "midrange"
        else:
            item["depth_category"] = "long range"
        
        return item
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return item

if __name__ == "__main__":
    bbox = (50, 50, 150, 150)
    for item in dataset:
        item = estimate_depth(item, bbox, pipe, output_dir, visualize=True)
        print(item)