from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import os
from PIL import Image
from datasets import load_dataset
import torch
import faiss

info = torch.load("image_embeddings.pt")

answers = info["answers"]
stored_embeddings = info["image_embeddings"]

stored_embeddings_numpy = [e.cpu().numpy() for e in stored_embeddings]

index0 = faiss.IndexFlatL2(stored_embeddings_numpy[0].shape[1])
index0.add(stored_embeddings_numpy[0])
index1 = faiss.IndexFlatL2(stored_embeddings_numpy[1].shape[1])
index1.add(stored_embeddings_numpy[1])
index2 = faiss.IndexFlatL2(stored_embeddings_numpy[2].shape[1])
index2.add(stored_embeddings_numpy[2])





import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    def extract_image_embedding(image):
        return process_images([image], image_processor, model.config)[0]

    def retrieve_similar_answer(image_path, image_tensor, top_k=1):
        embedding = image_tensor.view(1, -1).cpu().numpy()
        question_type = image_path.split("_")[-2]
        if question_type == "general":
            distances, indices = index0.search(embedding, k=top_k)
            return [answers[0][i] for i in indices[0]]
        if question_type == "regional":
            distances, indices = index1.search(embedding, k=top_k)
            return [answers[1][i] for i in indices[0]]
        if question_type == "suggestion":
            distances, indices = index2.search(embedding, k=top_k)
            return [answers[2][i] for i in indices[0]]


    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    
    results = {}
    for line in tqdm(questions):
        idx = line["id"]
        image_file = f"{line['id']}.png"
        qs = line["conversations"][0]["value"].replace("<image>", "")
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        # sample_answer = retrieve_similar_answer(image_path, image_tensor)[0]
        sample_answer = retrieve_similar_answer(image_path, image_tensor, top_k=3)
        # qs = qs + "\n\nResponse in the same way as the following reference response:\n" + sample_answer
        # qs = qs + "\n\nResponse in the same way as the following reference response while avoiding any direct information from the reference response.\n" + sample_answer
        # qs = qs + "\n\nThree example responses are provided. Choose one to generate response in the similar style which can answer the question best while avoiding any direct information from the selected example.\nResponse1:\n\"" + sample_answer[0] +"\"\n\nResponse2:\n\"" + sample_answer[1] + "\"\n\nResponse3:\n\"" + sample_answer[2] + "\"\n\n"
        # qs = qs + "\n\nThree example responses are provided below. You must choose one style to guide the structure of your answer. When generating your response, please follow these rules:\n1. Do not use any specific details, numbers, or descriptions from the examples.\n2. Ensure your answer is fully based on the provided image and question.\n3. Begin your response with a natural sentence start, and make sure the output is grammatically complete.\n\nReference 1:\n\"" + sample_answer[0] + "\"\n\nReference 2:\n\"" + sample_answer[1] + "\"\n\nReference 3:\n\"" + sample_answer[2] + "\"\n\nYour task: Generate a response based on the image and question using one of the styles provided."
        # qs = qs + "\n\nStep 1: Three example response styles are shown below. Choose one of these styles as a format for answering. You will use this style in Step 2.\n\nStyle 1:\n\"" + sample_answer[0] + "\"\n\nStyle 2:\n\"" + sample_answer[1] + "\"\n\nStyle 3:\n\"" + sample_answer[2] + "\"\n\nStep 2: Based on the image and the question, generate an answer in the chosen style. Make sure the answer is fully based on the image content and does not use any specific information from the examples provided."
        # qs = qs + "\n\nStep 1: Three example response styles are shown below. Review all three examples carefully. Choose the one that provides the most detailed and comprehensive answer to the question, and use it as the format for your answer in Step 2.\n\nStyle 1:\n\"" + sample_answer[0] + "\"\n\nStyle 2:\n\"" + sample_answer[1] + "\"\n\nStyle 3:\n\"" + sample_answer[2] + "\"\n\nStep 2: Based on the image and the question, generate an answer in the chosen style. Make sure your answer is fully based on the image content and does not use any specific information from the examples provided."
        qs = qs + "\n\nStep 1: Three example response styles are shown below. Review all three examples carefully. Choose the one that provides the most accurate information to the question, and use it as the format for your response in Step 2.\n\nStyle 1:\n\"" + sample_answer[0] + "\"\n\nStyle 2:\n\"" + sample_answer[1] + "\"\n\nStyle 3:\n\"" + sample_answer[2] + "\"\n\nStep 2: Based on the image and the question, generate an answer in the chosen style. Make sure your response is fully based on the image content, begin your response with a natural sentence and does not use any specific information from the examples provided."
        # print(qs)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        # ans_file.write(json.dumps({"question_id": idx,
        #                            "prompt": cur_prompt,
        #                            "text": outputs,
        #                            "answer_id": ans_id,
        #                            "model_id": model_name,
        #                            "metadata": {}}) + "\n")
        # ans_file.flush()
        print(idx, outputs)
        
        results[idx] = outputs
    # ans_file.close()
    with open(args.answers_file, 'w') as f:
        json.dump(results, f, indent=4) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
