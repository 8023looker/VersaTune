from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_ckpt = "../../models/baichuanSEED"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, padding_side = 'left', trust_remote_code = True)
dataset = [f"{tokenizer.eos_token}"] * 10

'''
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
'''


sampling_params = SamplingParams(n=1,
                                temperature=1.0,
                                top_p=1,
                                top_k=-1,
                                max_tokens=15999,
                                skip_special_tokens=True,
                                )

llm = LLM(model=model_ckpt, trust_remote_code=True)
outputs = llm.generate(dataset, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(output)