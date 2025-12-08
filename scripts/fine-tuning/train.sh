
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

rm -rf __pycache__
echo "__pycache__ removed"

EXPERIMENT_NAME="train_lr${LR}_wd${WEIGHT_DECAY}_t2iweight${T2I_WEIGHT}_fixed"

mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}

pip install torch torchvision ftfy regex tqdm wandb
pip install git+https://github.com/openai/CLIP.git

python -m src.clip.train.trainer  \
    --model_name ${MODEL_NAME} \
    --images_dir ${IMAGES_DIR} \
    --texts_dir ${TEXTS_DIR} \
    --splits_file "splits.json" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --t2i_weight ${T2I_WEIGHT} \
    --t2t_weight ${T2T_WEIGHT} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --grad_clip 1.0 \
    --mixed_precision \
    --num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name ${EXPERIMENT_NAME} \
    --mixed_precision \
    --seed 42
