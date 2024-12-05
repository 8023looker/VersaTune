DATASET_FOLDER_LIST=("baichuan2-7b" "baichuan2-13b" "llama-3-8B" "Qwen2-7B" "Qwen2.5-7B" "Qwen2.5-14B" "Qwen1.5-7B")
# DATASET_FOLDER_LIST=("Qwen2-7B" "Qwen2.5-7B" "Qwen2.5-14B" "Qwen1.5-7B")

for DATA_DIR in "${DATASET_FOLDER_LIST[@]}"; do
    echo $DATA_DIR
    # CUDA_VISIBLE_DEVICES=4,5,6,7 \

    python qwen_infer_domain.py $DATA_DIR
done