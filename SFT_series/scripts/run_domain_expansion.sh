#!/bin/bash
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# TOKENIZERS_PARALLELISM=false \
# NCCL_P2P_DISABLE=1 \

DATASET_LIST=(
              ../../../dataset/law/Lawyer-Instruct/alpacmygavel.json 
              ../../../dataset/finance/finance-alpaca/Cleaned_date.json 
              ../../../dataset/medical/medical_meadow_medqa/medical_meadow_medqa.json 
              ../../../dataset/science/sonnet3.5_science_conversations/sonnet3.5_science_conversations_sharegpt.jsonl 
              ../../../dataset/alpaca_rating/alpaca_data.json 
              ../../../dataset/code/codeAlpaca/code_alpaca_20k.json
            )
# MODEL=../../../model/llama-3-8B/
MODEL=../../../model/baichuan2-13b-hf/



for ((i=1; i<=1; i++))
do
    echo "It's $i epoch"

    for dataset in "${DATASET_LIST[@]}"; do
        echo $dataset
        output_dir="../output/domain_expansion_$(basename $MODEL)_$(basename $dataset)"

        # CUDA_VISIBLE_DEVICES=0,1,2,3 \
        
        torchrun --nproc_per_node=8 --master_port=4527 ../src/domain_expansion/train_domain_expansion.py \
            --model_name_or_path $MODEL \
            --data_path $dataset \
            --bf16 True \
            --output_dir $output_dir \
            --num_train_epochs 4 \
            --per_device_train_batch_size 4 \
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
            --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
            --tf32 True \
            --overwrite_output_dir True \

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