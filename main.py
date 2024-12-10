from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torchvision import transforms

# Load the dataset
train_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="train", streaming=True)
val_dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="val", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("LLaVA-1.5-7b")

def preprocess(example):
    '''
    Preprocess the example
    :param example: the example from the dataset
    :return: a dictionary containing the image, input_ids, and attention_mask
    '''  
    image = Image.open(example["image"]).convert("RGB")
    # data augmentation
    transfig = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    image = transfig(image)
    conversation = " ".join([turn["text"] for turn in example["conversations"]])
    tokenized = tokenizer(conversation, truncation=True, max_length=512, return_tensors="pt")

    return {
        "image": image,
        "input_ids": tokenized["input_ids"].squeeze(0),
        "attention_mask": tokenized["attention_mask"].squeeze(0),
    }


train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Define the model, only can use LLaVA-1.5-7b!!
model = AutoModelForCausalLM.from_pretrained("LLaVA-1.5-7b")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=8,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./llava-1.5-7b-finetuned")
tokenizer.save_pretrained("./llava-1.5-7b-finetuned")
