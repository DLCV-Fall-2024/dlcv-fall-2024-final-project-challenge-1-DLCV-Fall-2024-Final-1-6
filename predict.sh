python predict.py \
    --model-path checkpoints/llava-v1.5-7b-lora_2 \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file data/test/annotation.json \
    --image-folder data/test/images \
    --answers-file results/answer.json
