#!/bin/bash
# Zero-shot CLIP-base evaluation 


IMAGES_DIR="ArtKB/images"
TEXTS_DIR="ArtKB/texts"
OUTPUT_DIR="experiments/zeroshot"

mkdir -p $OUTPUT_DIR

python -m src.clip.eval.evaluator \
    --model_name "ViT-B/32" \
    --checkpoint "pretrained" \
    --images_dir $IMAGES_DIR \
    --texts_dir $TEXTS_DIR \
    --description_type "hybrid_o2" \
    --split "test" \
    --batch_size 64 \
    --device "cuda" \
    --output_file "$OUTPUT_DIR/clip_base_B32.json" \
    --seed 42


python -m src.clip.eval.evaluator \
    --model_name "ViT-L/14" \
    --checkpoint "pretrained" \
    --images_dir $IMAGES_DIR \
    --texts_dir $TEXTS_DIR \
    --description_type "hybrid_o2" \
    --split "test" \
    --batch_size 64 \
    --device "cuda" \
    --output_file "$OUTPUT_DIR/clip_baseL14.json" \
    --seed 42

