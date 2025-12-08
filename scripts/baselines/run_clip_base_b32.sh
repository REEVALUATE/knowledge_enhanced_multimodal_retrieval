#!/bin/bash
# Zero-shot CLIP-base evaluation 

IMAGES_DIR="../ArtKB/images"
TEXTS_DIR="../ArtKB/texts/texts"
OUTPUT_DIR="experiments/zeroshot"

mkdir -p $OUTPUT_DIR

pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

python -m src.clip.eval.evaluator \
    --model_name "ViT-B/32" \
    --images_dir $IMAGES_DIR \
    --texts_dir $TEXTS_DIR \
    --splits_file "../ArtKB/splits.json" \
    --split "test" \
    --batch_size 64 \
    --device "cuda" \
    --output_file "$OUTPUT_DIR/clip_base_b32.json" \
    --seed 42

