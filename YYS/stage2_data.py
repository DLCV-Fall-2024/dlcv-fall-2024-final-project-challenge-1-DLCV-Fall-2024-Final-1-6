import gc
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import json
import os
from datasets import load_dataset
from ultralytics import YOLO
from retrieval_YOLOCLIP import ImageRetrieval



def setup_yolo():
    """Initialize YOLO model"""
    model_path = "/home/jasper0314/YYS/DLCV-Fall-2024-Final-1-focusedroboticsforautonomousnavigation/YYS/yolo11x_1216.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
    return YOLO(model_path, task='detect')

def YOLO_inference(image, model):
    """Run YOLO inference on an image"""
    with torch.no_grad():
        results = model(image, stream=False)
        return [[result.names[result.boxes.cls[i].item()]
                for i in range(result.boxes.cls.shape[0])] for result in results]

def process_dataset(dataset, yolo_model, retriever, output_dir):
    """Process dataset with YOLO model and image retrieval"""
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []

    for idx, item in enumerate(tqdm(dataset)):
        # Get image and save it
        image = item["image"]
        image_filename = f"image_{idx}.jpg"
        
        # Save original image
        image.save(os.path.join(output_dir, image_filename))
        
        # Process with YOLO
        # yolo_results = YOLO_inference(image, yolo_model)
        
        # Get similar images using retriever
        similar_ids, ref_caption = retriever.retrieve(item["id"], image, k=2, category="general")
        
        # Create system and question prompts
        system_prompt = ("There is an image of traffic captured from the perspective of the ego car. "
                        "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), "
                        "vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, "
                        "directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, "
                        "miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the "
                        "seven categories above.")
        
        question_prompt = "Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
        
        # Create data entry
        data_entry = {
            "id": idx,
            "image": image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": (f"<image>\n{system_prompt}\n\n"
                            #  f"YOLO object detection shows: {yolo_results}\n\n" 
                             f"Example Response from a Similar Scene:\n{ref_caption}\n\n"
                             f"Question: {question_prompt}\n\n"
                             f"Answer:")
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
    
    # Initialize image retriever
    retriever = ImageRetrieval(
        max_objects=5,
        golden_json="stage2/golden.json",
        mapping_json="stage2/train_idmap.json"
    )
    
    # Process training data
    output_dir = "./stage2_train_general"
    process_dataset(dataset["train"], yolo_model, retriever, output_dir)
    
    # Process validation data
    output_dir = "./stage2_val_general"
    process_dataset(dataset["val"], yolo_model, retriever, output_dir)

if __name__ == "__main__":
    main()
