#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# 按照 SequentialSampler 进行训练
import os
os.environ["WANDB_MODE"] = "offline" # emmm 服务器连不上 wandb...
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Any, Union, List
import json
import random
import numpy as np

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer
import datasets

from loss_based_reweight import *

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.getcwd()) # 输出当前工作目录
import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    # for Alpaca dataset
    "prompt_input": ( # 拥有Input的情况
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": ( # 没有Input的情况
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "math_prompt": ( # gsm8k-ScRel
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    ),
    # vertical domain
    "code_prompt": ( # codeAlpaca
        "Below is an instruction that describes a code task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "code_prompt_no_input": ( # codeAlpaca
        "Below is an instruction that describes a code task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "finance_prompt": ( # finance-Alpaca
        "Below is an instruction that describes a financal task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "finance_prompt_no_input": ( # finance-Alpaca
        "Below is an instruction that describes a financal task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "law_prompt": ( # Lawyer-Instruct
        "Below is an instruction that describes a legal task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "law_prompt_no_input": ( # Lawyer-Instruct
        "Below is an instruction that describes a legal task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "medical_prompt": ( # medical-meadow-medqa
        "Below is an instruction that describes a medical task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "medical_prompt_no_input": ( # medical-meadow-medqa
        "Below is an instruction that describes a medical task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    # "science_prompt": ( # ScienceQA_text_only
    #     "Below is an instruction that describes a scientific task, paired with an input that provides further context. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    # )
    "sonnet3.5_science_conversations_prompt": ( # sonnet3.5_science_conversations (conversation 待修改)
        "Below is an instruction that describes a scientific task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )
    
    # "human_prompt": ( # shareGPT
    #     "Below is a conversation about science between a curious human and an artificial intelligence assistant. "
    #     # "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    #     "Provide a response that appropriately completes each request in this conversation.\n\n"
    #     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    # )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments_old:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class DataArguments:
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn_whole(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    
    
def _tokenize_fn(source_strings: Sequence[str], target_strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    """同时考虑了conversation: 对于source和target都是list的情况"""
    source_tokenized_list, target_tokenized_list = [], []
    for idx, string_or_list in enumerate(source_strings):
        if isinstance(string_or_list, list): # for conversation data
            source_tokenized_list.append([tokenizer(
                s,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for s in string_or_list])
            target_tokenized_list.append([tokenizer(
                t,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for t in target_strings[idx]])
        else: # for other data (math, code)
            source_tokenized_list.append(tokenizer(
                string_or_list,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ))
            target_tokenized_list.append(tokenizer(
                target_strings[idx],
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ))
    # debugging
    # print("source_tokenized_list", source_tokenized_list)
    # print("target_tokenized_list", target_tokenized_list)
    
    input_ids = labels = [tokenized[0].input_ids[0] if isinstance(tokenized, list) else tokenized.input_ids[0] for tokenized in source_tokenized_list]
    input_ids_lens = labels_lens = [
        utils.compute_conversation_source_input_len(tokenized, target_tokenized_list[sidx], tokenizer) if 
            isinstance(tokenized, list) else 
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for sidx, tokenized in enumerate(source_tokenized_list)
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str], # instruction + input
    targets: Sequence[str], # output
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [utils.concat_shareGPT_data(s, t) if isinstance(s, list) else s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn_whole(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, targets, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        if isinstance(source_len, list): # for shareGPT conversation data
            for cur_source_len in source_len:
                label[cur_source_len[0] : cur_source_len[0] + cur_source_len[1]] = IGNORE_INDEX
        else:
            label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path_list: List, tokenizer: transformers.PreTrainedTokenizer, domain_weight: Optional[Dict] = None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path_list, domain_weight) # domain weight can be float(ratio) or int(sample number)

        logging.warning("Formatting inputs...")
        
        # handle prompt for different dataset (待修改)
        # note: for "shareGPT" data, the source / target items are list format, while others are string format
        law_prompt, finance_prompt, medical_prompt, general_prompt, code_prompt = PROMPT_DICT["law_prompt"], PROMPT_DICT["finance_prompt"], PROMPT_DICT["medical_prompt"], PROMPT_DICT["prompt_input"], PROMPT_DICT["code_prompt"]
        law_prompt_no_input, finance_prompt_no_input, medical_prompt_no_input, general_prompt_no_input, code_prompt_no_input = PROMPT_DICT["law_prompt_no_input"], PROMPT_DICT["finance_prompt_no_input"], PROMPT_DICT["medical_prompt_no_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["code_prompt_no_input"]

        sources = list()
        for example in list_data_dict:
            if example["domain"] == "law":
                if example.get("input", "") == "":
                    sources.append(law_prompt_no_input.format_map(example))
                else:
                    sources.append(law_prompt.format_map(example))
            elif example["domain"] == "medical":
                if example.get("input", "") == "":
                    sources.append(medical_prompt_no_input.format_map(example))
                else:
                    sources.append(medical_prompt.format_map(example))
            elif example["domain"] == "finance":
                if example.get("input", "") == "":
                    sources.append(finance_prompt_no_input.format_map(example))
                else:
                    sources.append(finance_prompt.format_map(example))
            elif example["domain"] == "code":
                if example.get("input", "") == "":
                    sources.append(code_prompt_no_input.format_map(example))
                else:
                    sources.append(code_prompt.format_map(example))
            elif example["domain"] == "other":
                if example.get("input", "") == "":
                    sources.append(general_prompt_no_input.format_map(example))
                else:
                    sources.append(general_prompt.format_map(example))
            else: # science
                sources.append(utils.science_prompt_format_map(example))

        targets = [f"{example['output']}{tokenizer.eos_token}" if example["domain"] != "science" else utils.human_target_format_map(example, tokenizer) for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        
        self.source_string = [example for example in list_data_dict]
        self.data_index = list(range(len(self.input_ids)))
        
        # print("self.input_ids", self.input_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], source_string=self.source_string[i])

        
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # print("input_ids", input_ids)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            # source_string=source_string,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, batch_size, data_args, domain_weight) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_domain_weight = utils.eval_domain_quanity()
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=data_args.data_path, domain_weight=domain_weight)
    # 不同 domain 的数据集加载
    finance_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=[file_path for file_path in data_args.data_path if file_path.split('/')[2] == "finance-alpaca"], domain_weight=eval_domain_weight)
    law_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=[file_path for file_path in data_args.data_path if file_path.split('/')[2] == "Lawyer-Instruct"], domain_weight=eval_domain_weight)
    medical_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=[file_path for file_path in data_args.data_path if file_path.split('/')[2] == "medical_meadow_medqa"], domain_weight=eval_domain_weight)
    science_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=[file_path for file_path in data_args.data_path if file_path.split('/')[2] == "sonnet3.5_science_conversations"], domain_weight=eval_domain_weight)
    code_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=[file_path for file_path in data_args.data_path if file_path.split('/')[2] == "code_alpaca_20k"], domain_weight=eval_domain_weight)
    general_eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path_list=[file_path for file_path in data_args.data_path if file_path.split('/')[2] == "alpaca_rating"], domain_weight=eval_domain_weight)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer) # 其实后续没有用该 data_collator
    return dict(train_dataset=train_dataset, 
                finance_eval_dataset=finance_eval_dataset, 
                law_eval_dataset=law_eval_dataset, 
                medical_eval_dataset=medical_eval_dataset, 
                science_eval_dataset=science_eval_dataset, 
                general_eval_dataset=general_eval_dataset, 
                code_eval_dataset=code_eval_dataset,
                data_collator=data_collator)


# dataloader processing
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
def get_dataloader(dataset, data_collator, batch_size=1): # batch_size = 1, 1 batch contains 1 data point
    dataloader = DataLoader(dataset,
                            batch_size=batch_size, # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader


def evaluate_loss(domain_weight_dict, model_args, data_args, training_args):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map="auto", # for evaluation
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # 对于每一个 epoch 来说，都需要调整 domain_weight_dict
    data_module = make_supervised_data_module(tokenizer=tokenizer, batch_size=training_args.per_device_eval_batch_size * torch.cuda.device_count() * 4, data_args=data_args, domain_weight=domain_weight_dict)
    eval_dataloader = {
        "finance": get_dataloader(data_module["finance_eval_dataset"], data_module["data_collator"], batch_size=1), # * torch.cuda.device_count(), int(training_args.per_device_eval_batch_size / 2)
        "law": get_dataloader(data_module["law_eval_dataset"], data_module["data_collator"], batch_size=1), # * torch.cuda.device_count()
        "medical": get_dataloader(data_module["medical_eval_dataset"], data_module["data_collator"], batch_size=1), # * torch.cuda.device_count()
        "science": get_dataloader(data_module["science_eval_dataset"], data_module["data_collator"], batch_size=1),
        "code": get_dataloader(data_module["code_eval_dataset"], data_module["data_collator"], batch_size=1),
        "general": get_dataloader(data_module["general_eval_dataset"], data_module["data_collator"], batch_size=1)
    }
    reweight_trainer = ReweightTrainer(model=model, tokenizer=tokenizer, args=training_args)
    domain_weight_dict = reweight_trainer.loss_based_reweighting(eval_dataloader,
                                                                 model,
                                                                 prev_domain_weight=domain_weight_dict)
    model_params_dict = utils.read_model_param()
    model_params_dict["domain_weight"] = domain_weight_dict
    utils.write2model_param(model_params_dict)
    
    return domain_weight_dict


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_params_dict = utils.initialize_model_params()
    model_args.model_name_or_path = model_params_dict["model_path"]
    domain_weight_dict = model_params_dict["domain_weight"] # domain_weight_initialization (读入)
    
    domain_weight_dict = evaluate_loss(domain_weight_dict, model_args, data_args, training_args)
    model_params_dict["domain_weight"] = domain_weight_dict
    utils.write2model_param(model_params_dict)