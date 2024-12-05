from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import time
import torch
from tqdm import tqdm
import ujson
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from multiprocessing import Pool
import os
from pathlib import Path
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

@dataclass
class InferenceArguments:
    model_name_or_path: str
    number: int
    output_dir: str
    batch_size: int
    do_sample: bool
    max_length: int
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float

    
class Text_Generator:
    def __init__(self, idx: int, texts: Sequence[str], inference_args: InferenceArguments):
        super().__init__()
        self.idx = idx
        self.device = f"cuda:{idx}"
        self.inference_args = inference_args
        self.tokenizer = AutoTokenizer.from_pretrained(self.inference_args.model_name_or_path, padding_side = 'left', trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(self.inference_args.model_name_or_path, trust_remote_code = True).to(self.device)

        self.model.eval()
        # self.model.config.eos_token_id = -1
        # self.model.generation_config.eos_token_id = -1
        
        Path(self.inference_args.output_dir).mkdir(parents = True, exist_ok = True)
        now = datetime.now()
        self.formatted_time = now.strftime("%m_%d_%H_%M_%S")
        self.output_file = os.path.join(self.inference_args.output_dir, f"output_{self.idx}_{self.formatted_time}.jsonl")
        self.batch_size = self.inference_args.batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # prepare dataloader
        self.dataloader = torch.utils.data.DataLoader(texts, batch_size = self.batch_size, shuffle = False)
        
        
    def generate(self):
        with open(self.output_file, 'w') as f:
            with torch.no_grad():
                for text_batch in tqdm(self.dataloader):
                    
                    # add eos token 
                    text_batch = [f"{self.tokenizer.eos_token}{text}" for text in text_batch]
                        
                    inputs = self.tokenizer(text_batch, return_tensors="pt", padding = 'longest', add_special_tokens = False).to(self.device)
                    outputs = self.model.generate(**inputs,
                                                max_length = self.inference_args.max_length,
                                                do_sample = self.inference_args.do_sample,
                                                top_k = self.inference_args.top_k,
                                                top_p = self.inference_args.top_p,
                                                temperature = self.inference_args.temperature,
                                                repetition_penalty = self.inference_args.repetition_penalty,
                                                num_return_sequences=1)
                    try:
                        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
                    except IndexError as e:
                        print(f"Error: {e}")
                        for i, token_id in enumerate(outputs[0]):
                            try:
                                single_decoded = self.tokenizer.decode(token_id, skip_special_tokens = False)
                            except IndexError:
                                with open(f'error_{self.idx}_{self.formatted_time}.log', 'a') as ef:
                                    ef.write(f"Invalid token ID found: {token_id} at position {i}\n")
                        continue
                    assert len(text_batch) == len(generated_texts)
                    [f.write(ujson.dumps({'prompt': text_batch[i], 'response': generated_texts[i]}, ensure_ascii = False) + '\n') for i in range(len(text_batch))]
                

def worker(params: tuple):
    idx, dataset, inference_args = params
    print(f'worker {idx} start: dataset length {len(dataset)}')

    model = Text_Generator(idx, dataset, inference_args)
    
    model.generate()

def Infer():
    # parse arguments
    parser = transformers.HfArgumentParser(InferenceArguments)
    inference_args, = parser.parse_args_into_dataclasses()
    
    # prepare dataset
    dataset = [''] * inference_args.number
        
    # split tasks
    device_count = torch.cuda.device_count()
    work_load = (len(dataset) + device_count - 1) // device_count
    split_dataset = [dataset[i * work_load: (i + 1) * work_load] for i in range(device_count)]
    with Pool(processes = device_count) as pool:
        pool.map(worker, [(idx, piece, inference_args) for idx, piece in enumerate(split_dataset)])

if __name__ == '__main__':
    Infer()
