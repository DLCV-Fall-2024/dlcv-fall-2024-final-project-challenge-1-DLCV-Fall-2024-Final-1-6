import os
import sys
import json
import cv2

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from transformers import pipeline
from ultralytics import YOLO
from datasets import load_dataset

from number_to_words import number_to_words


# Constants
YOLO_MODEL_PATH = "weights/yolov8x-worldv2.pt"
DEPTH_MODEL_PATH = "depth-anything/Depth-Anything-V2-large-hf"

CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "van",
    "SUV",
    "trailer",
    "moped",
    "ambulance",
    "construction vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclists",
    "road users",
    "red traffic light",
    "traffic light",
    "parking sign",
    "warning traffic sign",
    "directional traffic sign",
    "traffic box",
    "sentry box",
    "traffic cone",
    "traffic island",
    "barrier",
    "bollard",
    "debris",
    "machinery",
    "dustbin",
    "concrete block",
    "dog",
    "chair",
    "phone booth",
    "streetlights"
]

BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_our_dataset(split):
    """Load the dataset for the given split."""
    return load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)


def custom_collate_fn(batch):
    """Custom collate function for DataLoader."""
    ids = [item["id"] for item in batch]
    images = [item["image"] for item in batch]
    conversations = [item["conversations"] for item in batch]

    return {"ids": ids, "images": images, "conversations": conversations}


def main(data_path, split_type="train"):
    assert split_type in ["train", "val", "test"], "invalid split type"
    # Create directories
    depth_image_path = os.path.join(data_path, "depth_maps", "png")
    depth_npy_path = os.path.join(data_path, "depth_maps", "npy")
    image_path = os.path.join(data_path, f"{split_type}_images")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(depth_image_path, exist_ok=True)
    os.makedirs(depth_npy_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    # Load dataset
    dataset = load_our_dataset(split_type)
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    final_output = []

    # Initialize model and pipeline
    data = dict()
    detection_model = YOLO(YOLO_MODEL_PATH)
    depth_pipe = pipeline("depth-estimation", model=DEPTH_MODEL_PATH, device=DEVICE)

    # Set custom classes if defined
    if CLASSES:
        detection_model.set_classes(CLASSES)

    # Execute prediction for specified categories on an image
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        yolo_results = detection_model.predict(batch["images"], conf=0.15)
        depth_results = depth_pipe(batch["images"])

        for index, (yolo_result, depth_result) in enumerate(
            zip(yolo_results, depth_results)
        ):
            image_name = batch["ids"][index]
            image_data = []
            
            # initialize category count
            category_count = dict()
            for category in CLASSES:
                category_count[category] = 0
                
            depth_map = np.array(depth_result["predicted_depth"])
            depth_map_tensor = torch.tensor(depth_map)
            
            w, h = batch["images"][index].size
            
            # to resize the depth map back to the original image shape
            depth_map = F.interpolate(
                depth_map_tensor[None, None], (h, w), mode="bilinear", align_corners=False
            )[0, 0]
            depth_map = depth_map.cpu().numpy()
            depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map_png = (depth_map_normalized * 255).astype(np.uint8) 

            # Save as .npy file
            np.save(os.path.join(depth_npy_path, f"{image_name}.npy"), depth_map_normalized)
                
            # Save as .png file
            cv2.imwrite(os.path.join(depth_image_path, f"{image_name}.png"), depth_map_png)


            for box in yolo_result.boxes:
                score = box.conf.item()
                label = box.cls.item()
                xyxy = box.xyxy[0].tolist()

                object_unique_name = (
                    detection_model.names.get(box.cls.item())
                    + " "
                    + number_to_words(
                        category_count[detection_model.names.get(box.cls.item())] + 1
                    )
                )
                category_count[detection_model.names.get(box.cls.item())] += 1

                image_data.append(
                    {
                        "category_name": detection_model.names.get(box.cls.item()),
                        "object_unique_name": object_unique_name,
                        "bbox_n": box.xyxyn[0].tolist(),
                        "bbox": xyxy,
                        "confidence": float(score),
                    }
                )

            # important
            data[image_name] = image_data

            # save the absolute path for image, depth map, and depth map in .npy format
            final_output.append(
                {
                    "id": image_name,
                    "image": os.path.abspath(os.path.join(image_path, f"{image_name}.png")),
                    "depth_png": os.path.abspath(os.path.join(depth_image_path, f"{image_name}.png")),
                    "depth_npy": os.path.abspath(os.path.join(depth_npy_path, f"{image_name}.npy")),
                    "conversations": batch["conversations"][index],
                    "bounding_box": image_data
                }
            )
            
            batch["images"][index].save(os.path.join(image_path, f"{image_name}.png"))
        
        break

    with open(os.path.join(data_path, f"{split_type}.json"), "w") as f:
        json.dump(final_output, f, indent=4)
        print(f"Saved to {os.path.join(data_path, f'{split_type}.json')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    # print