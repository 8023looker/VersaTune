#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 \

# --data_path ../../../dataset/law/Lawyer-Instruct/alpacmygavel.json ../../../dataset/finance/finance-alpaca/Cleaned_date.json ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl ../../../dataset/alpaca_rating/alpaca_data.json ../../../dataset/code/codeAlpaca/code_alpaca_20k.json \

for ((i=1; i<=1; i++))
do
    echo "It's $i epoch"

    torchrun --nproc_per_node=4 --master_port=4529 ../src/vertical_domain_evaluate.py \
        --model_name_or_path ../../../model/llama2-7b-hf/ \
        --data_path ../../../dataset/law/Lawyer-Instruct/alpacmygavel.json ../../../dataset/finance/finance-alpaca/Cleaned_date.json ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl ../../../dataset/alpaca_rating/alpaca_data.json ../../../dataset/code/codeAlpaca/code_alpaca_20k.json \
        --bf16 True \
        --output_dir ../output/loss-evaluate/ \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --fsdp "full_shard auto_wrap" \
        --report_to "none" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --overwrite_output_dir True \

done