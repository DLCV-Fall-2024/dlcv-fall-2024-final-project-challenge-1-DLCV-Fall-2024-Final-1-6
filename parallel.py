import os
import json
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset

DEPTH_MODEL_PATH = "depth-anything/Depth-Anything-V2-large-hf"
TRAIN_BATCH_SIZE = 5
AVAILABLE_DEVICES = [0, 1, 2, 3]

def load_our_dataset(split, rank=0, world_size=1):
    """Load dataset with sharding for parallel processing"""
    return load_dataset(
        "ntudlcv/dlcv_2024_final1", 
        split=split, 
        streaming=True
    ).shard(num_shards=world_size, index=rank)

def custom_collate_fn(batch):
    ids = [item["id"] for item in batch]
    images = [item["image"] for item in batch]
    conversations = [item["conversations"] for item in batch]
    return {"ids": ids, "images": images, "conversations": conversations}

def process_batch(rank):
    """Process batches on a single GPU"""
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    
    # Initialize model for this GPU
    depth_pipe = pipeline("depth-estimation", 
                         model=DEPTH_MODEL_PATH, 
                         device=device)
    
    # Load sharded dataset for this process
    dataset = load_our_dataset("test", rank=rank, world_size=len(AVAILABLE_DEVICES))
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Process batches
    for batch in tqdm(loader, desc=f"GPU {rank}"):
        depth_results = depth_pipe(batch["images"])
        
        for index, depth_result in enumerate(depth_results):
            image_name = batch["ids"][index]
            depth_map = np.array(depth_result["predicted_depth"])
            depth_map_tensor = torch.tensor(depth_map, device=device)
            
            w, h = batch["images"][index].size
            depth_map = F.interpolate(
                depth_map_tensor[None, None], 
                (h, w), 
                mode="bilinear", 
                align_corners=False
            )[0, 0]
            
            depth_map = depth_map.cpu().numpy()
            depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map_png = (depth_map_normalized * 255).astype(np.uint8)
            
            # Save outputs
            cv2.imwrite(f"/home/twszmxa453/CVPR2025/dlcv_test_data/depth_maps/png/{image_name}.png", depth_map_png)
            np.save(f"/home/twszmxa453/CVPR2025/dlcv_test_data/depth_maps/npy/{image_name}.npy", depth_map_normalized)
            batch["images"][index].save(f"/home/twszmxa453/CVPR2025/dlcv_test_data/test_images/{image_name}.png")

def main():
    """Main function to start parallel processing"""
    # Create output directories
    os.makedirs("/home/twszmxa453/CVPR2025/dlcv_test_data/depth_maps/png", exist_ok=True)
    os.makedirs("/home/twszmxa453/CVPR2025/dlcv_test_data/depth_maps/npy", exist_ok=True)
    os.makedirs("/home/twszmxa453/CVPR2025/dlcv_test_data/test_images", exist_ok=True)
    
    # Start processes
    processes = []
    for rank in AVAILABLE_DEVICES:
        p = mp.Process(target=process_batch, args=(rank,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()