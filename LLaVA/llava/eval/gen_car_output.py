import os
import json
import argparse
import copy
import re
import requests
from io import BytesIO
import random

import cv2
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import transformers
from datasets import load_dataset
from PIL import Image
from typing import Dict
from scipy.ndimage import zoom

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import CLASSES, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_and_process_dataset(data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args):
    if data_path.startswith('/mnt/'):
        dataset = load_dataset('json', data_files=data_path)
    else:
        dataset = load_dataset(data_path)
    
    train_data = dataset['train']
    max_samples = 0
    if max_samples > 0:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
    
    return train_data, tokenizer, data_args

def get_lengths(train_data):
    length_list = []
    for sample in train_data:
        img_tokens = 128 if 'image' in sample else 0
        length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    return length_list

def visualize_bbox_map(additional_info, path):
    for i in range(np.array(additional_info).shape[-1]):
        if not np.array_equal(additional_info[..., i], np.zeros_like(additional_info[..., i])):
            plt.imshow(additional_info[..., i], cmap='viridis')
            plt.colorbar()
            plt.savefig(f'llava/train/visualize_images/{path}')
            plt.close()

def generate_bbox_map(pil_img, bboxes, categories):
    w, h = pil_img.size
    num_categories = len(categories)
    bbox_map = np.zeros((h, w, num_categories), dtype=np.uint8)
    
    category_to_idx = {category: idx for idx, category in enumerate(categories)}

    for bbox_dict in bboxes:
        bbox = bbox_dict['bbox']
        category_name = bbox_dict['category_name']

        if category_name not in categories:
            continue

        channel_idx = category_to_idx[category_name]
        x_min, y_min, x_max, y_max = map(round, bbox)
        x_min, x_max = np.clip([x_min, x_max], 0, w - 1)
        y_min, y_max = np.clip([y_min, y_max], 0, h - 1)
        bbox_map[y_min:y_max+1, x_min:x_max+1, channel_idx] = 1

    return bbox_map

def preprocess_image(image, processor, sources):
    assert len(image) == 1
    image = image[0]
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')

    bbox_map = generate_bbox_map(image, sources['bounding_box'])
    depth_map = np.load(sources["depth_npy"])
    depth_map = np.expand_dims(depth_map, axis=2)
    additional_info = np.concatenate((bbox_map, depth_map), axis=2)
    additional_info = expand2square_np(additional_info)
    additional_info = preprocess_additional_info(additional_info)
    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    print("image shape:", image.shape)
    # print("image size:", image.size())
    additional_info = np.transpose(additional_info, (2, 0, 1))
    image = torch.cat([image, torch.from_numpy(additional_info)], dim=0).unsqueeze(0)

    print(image.shape)
    
    return image
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def expand2square_np(img_np):
    h, w, c = img_np.shape
    square_size = max(h, w)
    expanded_img = np.zeros((square_size, square_size, c))
    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    expanded_img[y_offset:y_offset+h, x_offset:x_offset+w, :] = img_np
    return expanded_img

def preprocess_additional_info(additional_data, size=336, do_resize=True, crop_size=336):
    h, w, c = additional_data.shape
    resized_additional_data = np.zeros((size, size, c))
    if do_resize:
        scaling_factor = (size / h, size / w)
        for i in range(c):
            resized_additional_data[:, :, i] = cv2.resize(additional_data[:, :, i], (size, size), interpolation=cv2.INTER_NEAREST)
    if crop_size:
        top = (size - crop_size) // 2
        left = (size - crop_size) // 2
        resized_additional_data = resized_additional_data[top:top+crop_size, left:left+crop_size, :]
    return resized_additional_data



def generate_bbox_map(pil_img, bboxes, categories = CLASSES):
    w, h = pil_img.size
    num_categories = len(categories)
    bbox_map = np.zeros((h, w, num_categories), dtype=np.uint8)
    
    category_to_idx = {category: idx for idx, category in enumerate(categories)}

    for bbox_dict in bboxes:
        bbox = bbox_dict['bbox']
        category_name = bbox_dict['category_name']

        if category_name not in categories:
            continue

        # Get the corresponding channel index for the category
        channel_idx = category_to_idx[category_name]

        # Convert bbox to integers and ensure they are within bounds
        x_min, y_min, x_max, y_max = map(round, bbox)
        x_min, x_max = np.clip([x_min, x_max], 0, w - 1)
        y_min, y_max = np.clip([y_min, y_max], 0, h - 1)

        # Set the pixels in the bbox to 1 for the corresponding channel
        bbox_map[y_min:y_max+1, x_min:x_max+1, channel_idx] = 1

    return bbox_map


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def preprocess_multistage_inference(sources, inference_data, raw_path):
    def get_task_type_from_id(image_id):
        """Extract task type from image_id (e.g., 'test_general_01' -> 'general')"""
        parts = image_id.split('_')
        if len(parts) >= 2:
            return parts[1]  # 'general', 'suggestion', etc.
        return None

    def load_data_from_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    multistage_sources = copy.deepcopy(sources)
    target_id = multistage_sources[0]['id']
    task_type = get_task_type_from_id(target_id)
    assert task_type in ["general", "regional", "suggestion"], "Invalid task type"
    if task_type == 'general' or task_type == 'regional':
        return multistage_sources
    
    completed_data = copy.deepcopy(inference_data)

    for item in completed_data:
        if not item:
            continue
        id_ = item.get("id", "").split('_')[-1]
        if target_id.split('_')[-1] == id_ and item.get("id", "").split('_')[1] == 'general':
            first_stage_QA = 'Here is the given knowledge of the image:\n\n' + item['conversations'][1]['value'] + '\n\n' + 'Now, please answer the following question based on the given knowledge.\n\n'
            # first_stage_QA = 'Here are the given knowledge of an QA pairs: Questions: ' + item['conversations'][0]['value'].split('<image>\n')[-1] + '\nAnswers: ' + item['conversations'][1]['value']
            multistage_sources[0]['conversations'][0]['value'] = multistage_sources[0]['conversations'][0]['value'].replace('<image>\n', '<image>\n' + first_stage_QA)
    return multistage_sources
    
def eval_model(args):

    annotations = None
    with open(args.annotation_file, "r") as f:
        annotations = json.load(f)
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    qs_dict = {
        "general": "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.",
        "suggestion": "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.",
        "regional": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
    }
    for k, qs in qs_dict.items():
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        qs_dict[k] = qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    for k, qs in qs_dict.items():
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        qs_dict[k] = prompt

    all_outputs = {}
    general_outputs = {}
    for annotation in tqdm(annotations):
        task_type = annotation["id"].split("_")[1]
        if task_type == 'suggestion':
            continue
        print("task_type", task_type)
        # skip suggestion
        image_file = annotation["image"]
        assert os.path.exists(image_file), f"Image file {image_file} does not exist."
        image = [load_image(image_file)]
        print("image_file:", image_file)
        image_size = [image[0].size]
        image_tensor = preprocess_image(
            image,
            image_processor,
            annotation
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(qs_dict[task_type], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        all_outputs[image_file.split('/')[-1].split('.')[0]] = outputs
        print(outputs)
        if task_type == 'general':
            general_outputs[annotation["id"].split("_")[2]] = outputs

    for annotation in tqdm(annotations):
        task_type = annotation["id"].split("_")[1]
        if task_type == 'suggestion':
            first_stage_QA = 'Here are the given knowledge of an QA pairs: Questions: ' + qs_dict["general"] + '\nAnswers: ' + general_outputs[annotation["id"].split("_")[2]]
        else:
            continue
        print("task_type", task_type)
        # add region answers to suggestion
        image_file = annotation["image"]
        assert os.path.exists(image_file), f"Image file {image_file} does not exist."
        image = [load_image(image_file)]
        print("image_file:", image_file)
        image_size = [image[0].size]
        image_tensor = preprocess_image(
            image,
            image_processor,
            annotation
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(first_stage_QA + qs_dict[task_type], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        all_outputs[image_file.split('/')[-1].split('.')[0]] = outputs
        print(outputs)


    with open(args.output_file, "w") as f:
        json.dump(all_outputs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="submission.json")
    parser.add_argument("--annotation_file", type=str, default="/mnt/HDD_1/walker/dlcv_json_files/test.json")
    args = parser.parse_args()
    eval_model(args)
