# base version of qwen_infer using vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import ujson

# qwen 72b 需要 4 卡
import os
os.environ["CUDA_VISIBLE_DEVICES"]=4,5,6,7

model_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/liyouquan/huggingface_models/Qwen2.5-72B-Instruct/"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512) # max_tokens=512

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, tensor_parallel_size=4)

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")