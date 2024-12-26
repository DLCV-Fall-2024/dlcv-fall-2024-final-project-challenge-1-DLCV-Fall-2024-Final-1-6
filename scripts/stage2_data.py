import gc
from weakref import ref
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import json
import os
import argparse
from datasets import load_dataset
from retrieval_YOLOCLIP import ImageRetrieval


def parse_args():
    parser = argparse.ArgumentParser(description='Process images with YOLO and retrieval')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str,
                      default='ntudlcv/dlcv_2024_final1',
                      help='Name of the dataset to load')
    parser.add_argument('--split', type=str,
                      choices=['train', 'val', 'test'],
                      default='test',
                      help='Dataset split to process')
    parser.add_argument('--num_proc', type=int,
                      default=4,
                      help='Number of processes for dataset loading')
    
    # Retrieval arguments
    parser.add_argument('--yolo_model', type=str,
                      default='checkpoint/yolo11x.pt',
                      help='Path to YOLO model checkpoint')
    parser.add_argument('--max_objects', type=int,
                      default=5,
                      help='Maximum number of objects for retrieval')
    parser.add_argument('--golden_json', type=str,
                      default='stage2/golden.json',
                      help='Path to golden JSON file')
    parser.add_argument('--mapping_json', type=str,
                      default='stage2/train_idmap.json',
                      help='Path to mapping JSON file')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                      default='./stage2_test_general',
                      help='Directory to save processed data')
    parser.add_argument('--save_images', action='store_true',
                      help='Whether to save original images')
    parser.add_argument('--save_interval', type=int,
                      default=100,
                      help='Number of samples to process before saving')
    
    # Reference files
    parser.add_argument('--test_idmap', type=str,
                      default='stage2/test_idmap.json',
                      help='Path to test ID mapping file')
    parser.add_argument('--responses_file', type=str,
                      default='stage1_test_general/responses_new.json',
                      help='Path to responses JSON file')
    
    return parser.parse_args()


def extract_example_response(text):
    start_marker = "Example Response from a Similar Scene:"
    end_marker = "Question:"
    
    try:
        start_idx = text.index(start_marker) + len(start_marker)
        end_idx = text.index(end_marker)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return None
    
def extract_global_description(text):
    start_marker = "Example Response from a Similar Scene:\n\""
    end_marker = "\n\nQuestion:"
    
    try:
        start_idx = text.index(start_marker) + len(start_marker)
        end_idx = text.index(end_marker)
        return text[start_idx:end_idx].strip()
    except ValueError:
        return None

def process_dataset(dataset, retriever, args, test=True):
    """Process dataset with YOLO model and image retrieval"""
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    processed_data = []
    
    with open(args.test_idmap, "r") as f:
        idmap = json.load(f)
    with open(args.responses_file, "r") as f:
        responses = json.load(f)

    for idx, item in enumerate(tqdm(dataset)):
        image = item["image"]
        image_filename = os.path.join("images", f"image_{idx}.jpg")
        
        if args.save_images:
            image.save(os.path.join(args.output_dir, image_filename))
        
        # Get similar images using retriever
        similar_ids, ref_caption = retriever.retrieve(item["id"], image, k=2, category="general")
        
        if "0" in responses: # Old version
            if "regional" in item["id"]:
                image_id = idmap[item["id"]].split("_")[-1]
            else:
                image_id = item["id"].split("_")[-1]
        else: # New version
            if "regional" in item["id"]:
                image_id = idmap[item["id"]]
            else:
                image_id = item["id"].replace("suggestion", "general")
        
        if image_id not in responses:
            global_description = "No global description available."
        else:
            global_description = responses[image_id]
        
        # Determine question type
        if "general" in item["id"]:
            question_type = 0
        elif "regional" in item["id"]:
            question_type = 1
        else:
            question_type = 2
        
        # Create system and question prompts
        system_prompt = ("There is an image of traffic captured from the perspective of the ego car. "
                        "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), "
                        "vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, "
                        "directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, "
                        "miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the "
                        "seven categories above.")
        
        question_prompts = ["Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.",
                          "Please describe the object inside the red rectangles (bounding boxes) in the image and explain why it affect ego car driving.",
                          "Please provide driving suggestions for the ego car based on the current scene."]

        # Create data entry
        if test:
            data_entry = {
                "id": idx,
                "realid": item["id"],
                "image": image_filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": (f"<image>\n{system_prompt}\n\n"
                                f"Global Description: \n\"{global_description}\"\n\n"
                                f"Example Response from a Similar Scene:\n\"{ref_caption}\"\n\n"
                                f"Question: {question_prompts[question_type]}\n\n"
                                f"Answer:")
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
                        "value": (f"<image>\n{system_prompt}\n\n"
                                f"Global Description: \n\"{global_description}\"\n\n"
                                f"Example Response from a Similar Scene:\n\"{ref_caption}\"\n\n"
                                f"Question: {question_prompts[question_type]}\n\n"
                                f"Answer:")
                    },
                    {
                        "from": "gpt",
                        "value": item["conversations"][1]["value"]
                    }
                ]
            }
        processed_data.append(data_entry)
        
        if (idx + 1) % args.save_interval == 0:
            with open(os.path.join(args.output_dir, "processed_data.json"), 'w') as f:
                json.dump(processed_data, f, indent=4)
            gc.collect()

    # Final save
    with open(os.path.join(args.output_dir, "processed_data.json"), 'w') as f:
        json.dump(processed_data, f, indent=4)


def main():
    args = parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, num_proc=args.num_proc)
    
    # Initialize image retriever (RAG is always enabled)
    retriever = ImageRetrieval(
        max_objects=args.max_objects,
        yolo_model=args.yolo_model,
        golden_json=args.golden_json,
        mapping_json=args.mapping_json
    )
    
    # Process data
    is_test = args.split == 'test'
    process_dataset(dataset[args.split], retriever, args, test=is_test)


if __name__ == "__main__":
    main()
