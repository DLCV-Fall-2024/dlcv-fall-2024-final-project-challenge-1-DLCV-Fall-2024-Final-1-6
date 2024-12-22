from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import os
from PIL import Image
from datasets import load_dataset
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


disable_torch_init()
ds = load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True)
model_path = os.path.expanduser("checkpoints/llava-v1.5-7b-lora_2")
model_base = "lmsys/vicuna-7b-v1.5"
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
def extract_image_embedding(image):
    return process_images([image], image_processor, model.config)[0]
# assert False
MAX_NUM_IMAGES = 30000
dataset_iter = iter(ds)
answers = [[] for _ in range(3)]
stored_embeddings = [[] for _ in range(3)]
for i in range(MAX_NUM_IMAGES):
    try:
        item = next(dataset_iter)
        id_, conversation, image = item["id"], item["conversations"], item["image"]
        question_type = id_.split("_")[1]
        question = conversation[0]["value"]
        answer = conversation[1]["value"]
        image_embedding = extract_image_embedding(image).view(-1)
        if question_type == "general":
            answers[0].append(answer)
            stored_embeddings[0].append(image_embedding)
            # print(answers[0])
            # print(stored_embeddings[0][0].shape)
            # assert False
        elif question_type == "regional":
            answers[1].append(answer)
            stored_embeddings[1].append(image_embedding)
        elif question_type == "suggestion":
            answers[2].append(answer)
            stored_embeddings[2].append(image_embedding)
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        # print(image_embedding.shape)
        # assert False
    except StopIteration:
        break

image_embeddings = [torch.stack(stored_embeddings[i]) for i in range(3)]

torch.save({"answers": answers, "image_embeddings": image_embeddings}, "image_embeddings.pt")
