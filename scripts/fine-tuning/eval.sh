
# Set paths
IMAGES_DIR="../ArtKB/images"
TEXTS_DIR="../ArtKB/texts/texts"
OUTPUT_DIR="experiments/fine-tuning"

# Training configuration
MODEL_NAME="ViT-L/14"
BATCH_SIZE=64
NUM_EPOCHS=20
LR=5e-6
WEIGHT_DECAY=0.02
EARLY_STOPPING_PATIENCE=5
T2I_WEIGHT=0.7
T2T_WEIGHT=0.3

EXPERIMENT_NAME="train_lr${LR}_wd${WEIGHT_DECAY}_t2iweight${T2I_WEIGHT}"
# Create output directories
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}



pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# multi mode evaluation
python -m src.clip.eval.evaluator \
    --model_name "ViT-L/14" \
    --checkpoint "$OUTPUT_DIR/$EXPERIMENT_NAME/checkpoint_best.pt" \
    --images_dir $IMAGES_DIR \
    --texts_dir $TEXTS_DIR \
    --split "test" \
    --splits_file "splits.json" \
    --batch_size 64 \
    --device "cuda" \
    --output_file "$OUTPUT_DIR/$EXPERIMENT_NAME/eval.json" \
    --seed 42
