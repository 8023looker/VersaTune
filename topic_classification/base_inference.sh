#!/bin/bash

declare -a model_paths=(
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/llama2-7b-hf/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/Qwen1.5-7B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/Qwen2-7B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/Qwen2.5-7B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/llama2-13b-hf/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/Qwen2.5-14B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/model/llama-3-8B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Mistral-7B-v0.3/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Qwen2.5-14B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/llama2-7b-hf/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Qwen1.5-7B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Qwen2-7B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Qwen2.5-7B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/llama2-13b-hf/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/Qwen2.5-14B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/model/llama-3-8B/"
)

declare -a output_dirs=(
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/llama2-7b/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen1.5-7B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2-7B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2.5-7B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/llama2-13b/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2.5-14B/"
    # "/cpfs/29f69eb5e2e60f26/user/sft_intern/keerlu/CPT_params/topic_classification/model_output/llama-3-8B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/Mistral-7B-v0.3/" 
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2.5-14B/" 
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/llama2-7b/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen1.5-7B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2-7B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2.5-7B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/llama2-13b/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/Qwen2.5-14B/"
    "/cpfs/29f69eb5e2e60f26/code/sft_intern/keerlu/CPT_params/topic_classification/model_output/llama-3-8B/"
)

if [ "${#model_paths[@]}" -ne "${#output_dirs[@]}" ]; then
    echo "Error: The number of model paths and output directories must be the same."
    exit 1
fi


for (( i=0; i<${#model_paths[@]}; i++ )); do
    model=${model_paths[$i]}
    output=${output_dirs[$i]}
    
    python sequence_generation.py \
        --model_name_or_path "$model" \
        --number 10000\
        --output_dir "$output" \
        --batch_size 1 \
        --do_sample True \
        --max_length 16000 \
        --top_k 0 \
        --top_p 1.0 \
        --temperature 1.0 \
        --repetition_penalty 1.1
done

