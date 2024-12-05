# DATASET_LIST=(../../../dataset/law/Lawyer-Instruct/alpacmygavel.json ../../../dataset/finance/finance-alpaca/Cleaned_date.json ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl ../../../dataset/alpaca_rating/alpaca_data.json ../../../dataset/code/codeAlpaca/code_alpaca_20k.json)
DATASET_LIST=(../../../dataset/code/codeAlpaca/code_alpaca_20k.json)
MODEL_DIR=../../../model/Sheared-LLaMA-1.3B/

for DATA_DIR in "${DATASET_LIST[@]}"; do
    echo $DATA_DIR
    OUTPUT_DIR="../output/proxy_$(basename $DATA_DIR)_1.3B"
    CUDA_VISIBLE_DEVICES=0,1,2,3 \

    torchrun --nproc_per_node=4 --master_port=4585 ../src/train_proxy_loss.py \
        --model_name_or_path ${MODEL_DIR} \
        --data_path ${DATA_DIR} \
        --bf16 True \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 100 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --fsdp "full_shard auto_wrap offload" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --overwrite_output_dir True \
        --gradient_checkpointing True \
        # --seed 42 # random seed

done