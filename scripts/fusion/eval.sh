
T2I_WEIGHT=0.5
T2T_WEIGHT=0.5


# Set paths
IMAGES_DIR="../ArtKB/images"
TEXTS_DIR="../ArtKB/texts/texts"
OUTPUT_DIR="experiments/fine-tuning"


EXPERIMENT_NAME="train_lr5e-6_wd0.02_t2iweight0.7"
# Create output directories

pip install torch torchvision ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# multi mode evaluation
python -m src.clip.eval.evaluator_baseline \
    --model_name "ViT-L/14" \
    --checkpoint "$OUTPUT_DIR/$EXPERIMENT_NAME/checkpoint_best.pt" \
    --images_dir $IMAGES_DIR \
    --texts_dir $TEXTS_DIR \
    --split "test" \
    --splits_file "splits.json" \
    --batch_size 64 \
    --device "cuda" \
    --output_file "experiments/2-fusion/baseline_${T2I_WEIGHT}_${T2T_WEIGHT}_cpu.json" \
    --t2i_weight $T2I_WEIGHT \
    --t2t_weight $T2T_WEIGHT