#!/bin/bash
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# TOKENIZERS_PARALLELISM=false \
# NCCL_P2P_DISABLE=1 \

DATASET_LIST=(../../../dataset/law/Lawyer-Instruct/alpacmygavel.json 
              ../../../dataset/finance/finance-alpaca/Cleaned_date.json 
              ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json 
              ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl 
              ../../../dataset/alpaca_rating/alpaca_data.json 
              ../../../dataset/code/codeAlpaca/code_alpaca_20k.json)
MODEL_LIST=(
            # ../../../model/llama2-7b-hf/ 
            # ../../../model/llama-3-8B/ 
            # ../../../model/Qwen2.5-7B/ 
            # ../../../model/Qwen2-7B/ 
            # ../../../model/Qwen1.5-7B/)
            ../../../model/llama2-13b-hf/
            ../../../model/Qwen2.5-14B/)


for MODEL_DIR in "${MODEL_LIST[@]}"; do
    echo $MODEL_DIR
    MODEL_NAME="$(basename $MODEL_DIR)"
    OUTPUT_DIR="../output/multi_$(basename $MODEL_DIR)_versatune_dynamic"
    DECODER_LAYER="LlamaDecoderLayer"

    if [[ "$MODEL_NAME" == *"llama"* ]]; then
        echo "Using LlamaDecoderLayer for model: $MODEL_NAME"
        DECODER_LAYER="LlamaDecoderLayer"
    elif [[ "$MODEL_NAME" == *"Qwen"* ]]; then
        echo "Using Qwen2DecoderLayer for model: $MODEL_NAME"
        DECODER_LAYER="Qwen2DecoderLayer"
    elif [[ "$MODEL_NAME" == *"Mistral"* ]]; then
        echo "Using MistralDecoderLayer for model: $MODEL_NAME"
        DECODER_LAYER="MistralDecoderLayer"
    else
        echo "Unknown model type: $MODEL_NAME. Please specify a valid model directory."
        exit 1
    fi

    for ((i=1; i<=4; i++)); do
        echo "It's $i epoch"

        torchrun --nproc_per_node=8 --master_port=4585 ../src/train_topic_ratio.py \
            --model_name_or_path ${MODEL_DIR} \
            --data_path ../../../dataset/law/Lawyer-Instruct/alpacmygavel.json ../../../dataset/finance/finance-alpaca/Cleaned_date.json ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl ../../../dataset/alpaca_rating/alpaca_data.json ../../../dataset/code/codeAlpaca/code_alpaca_20k.json \
            --bf16 True \
            --output_dir ${OUTPUT_DIR} \
            --num_train_epochs 1 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 8 \
            --eval_strategy "no" \
            --save_strategy "epoch" \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap offload" \
            --report_to "none" \
            --fsdp_transformer_layer_cls_to_wrap "${DECODER_LAYER}" \
            --tf32 True \
            --overwrite_output_dir True \
            # --seed 42 # random seed

        if [ $i -ne 4 ]; then

            torchrun --nproc_per_node=4 --master_port=4529 ../src/vertical_domain_evaluate.py \
                --model_name_or_path ${MODEL_DIR} \
                --data_path ../../../dataset/law/Lawyer-Instruct/alpacmygavel.json ../../../dataset/finance/finance-alpaca/Cleaned_date.json ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl ../../../dataset/alpaca_rating/alpaca_data.json ../../../dataset/code/codeAlpaca/code_alpaca_20k.json \
                --bf16 True \
                --output_dir ../output/loss-evaluate/ \
                --num_train_epochs 1 \
                --per_device_train_batch_size 2 \
                --per_device_eval_batch_size 4 \
                --gradient_accumulation_steps 8 \
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
                --fsdp_transformer_layer_cls_to_wrap "${DECODER_LAYER}" \
                --tf32 True \
                --overwrite_output_dir True \

        fi
        
    done

done

# --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
# --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer'
# --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer'

# 13b
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 8 \

# 7b
# --per_device_train_batch_size 8 \
# --per_device_eval_batch_size 8 \
# --gradient_accumulation_steps 16 \