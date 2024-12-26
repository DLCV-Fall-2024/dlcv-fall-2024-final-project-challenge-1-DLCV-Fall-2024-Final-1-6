import sys
import argparse
# sys.path.append("./LLaVA")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model import LlavaLlamaForCausalLM
from peft import PeftModel
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
from llava.mm_utils import process_images, tokenizer_image_token
import json
import time
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='LLaVA Inference Script')
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b",
                        help='Path to the base model')
    parser.add_argument('--lora_path', type=str, default="checkpoint/stage1_lora_2/checkpoint-1400",
                        help='Path to the LoRA weights')
    parser.add_argument('--data_dir', type=str, default="stage1_test_general",
                        help='Directory containing the test data')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--min_new_tokens', type=int, default=30,
                        help='Minimum number of new tokens to generate')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save responses to file every N iterations')
    parser.add_argument('--output_json', type=str, default="stage1_test_general/responses.json")
    parser.add_argument('--input_json', type=str, default="stage1_test_general/processed_data.json")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

    # Load model and tokenizer
    model_name = get_model_name_from_path(args.model_path)
    # Add this line before loading the model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        None,
        model_name
    )
    model = PeftModel.from_pretrained(model, args.lora_path)

    # Setup paths and load data
    responses_file = args.output_json
       
    try:
        with open(responses_file, 'r') as f:
            responses = json.load(f)
        print(f"Loaded {len(responses)} existing responses from {responses_file}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"No valid existing responses file found, starting with empty responses")
        responses = {}

    with open(args.input_json) as f:
        datas = json.load(f)

    print(f"Loaded {len(datas)} examples")

    # Process each example
    for data in tqdm(datas):
        if str(data["id"]) in responses:
            print(f"Skipping {data['id']} - already processed")
            continue

        image = Image.open(os.path.join(args.data_dir, data["image"]))
        image_tensor = process_images([image], image_processor, model.config).to("cuda")
        prompt = data["conversations"][0]["value"]
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            torch.cuda.synchronize()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                use_cache=True
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Save the new response
        if "realid" in data:
            responses[data["realid"]] = outputs
        else:
            responses[data["id"]] = outputs
        
        # Save to file at specified intervals
        if len(responses) % args.save_interval == 0:
            with open(responses_file, 'w') as f:
                json.dump(responses, f, indent=4)

    # Final save
    with open(responses_file, 'w') as f:
        json.dump(responses, f, indent=4)

if __name__ == "__main__":
    main()