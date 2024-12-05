import json
import ujson
import numpy as np
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import RobertaModel
from accelerate import Accelerator, skip_first_batches
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer

def prepare_batch(batch, device=torch.device("cuda:0")): # device=torch.device("cuda:0")
    """ Move the batch to the device. """
    for key in batch: # 1 batch contains 1 item
        batch[key] = batch[key].to(device)

def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))
        
class ReweightTrainer(Trainer):
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)
        
    def grad_based_reweighting(self, dataloader,
                                model,
                                prev_domain_weight=None):

        torch.random.manual_seed(0)  # set the random seed for torch

        device = next(model.parameters()).device
        print("Device:", device)
        
        domain_grad_dict = {}
        for domain in dataloader.keys():
            domain_grad_dict[domain] = []
            
            cur_data_loader = dataloader[domain]
            for batch in tqdm(cur_data_loader, total=len(cur_data_loader)):
                # print("batch", batch) # keys: "input_ids", "attention_mask", "labels"
                prepare_batch(batch, device=device)
                domain_grad_dict[domain].append(self.obtain_gradients(model, batch))
            stacked_tensor = torch.stack(domain_grad_dict[domain])
            domain_grad_dict[domain] = torch.mean(stacked_tensor, dim=0)
            # torch.cuda.empty_cache()
        
        domain_weight_dict = self.domain_reweighting(domain_grad_dict, prev_domain_weight)
        print("domain_weight_dict", domain_weight_dict)
        model.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Finished")
        return domain_weight_dict
        
    def obtain_gradients(self, model, batch): # put all tensors on the same device
        """ obtain gradients. """
        loss = model(**batch).loss
        # loss.backward()
        self.accelerator.backward(loss)
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if name == "lm_head.weight":
        #             print("name", name, param.grad)
                # elif name == "model.norm.weight":
                #     print("name", name, param.grad)
        
        full_grad_concat = None
        for name, param in model.named_parameters():
            if param.grad is not None and name == "model.norm.weight": # dim of "lm_head.weight": 131076096, dim of "model.norm.weight": 4096
                flat_grad = param.grad.detach().flatten()
                # add to full grad
                if full_grad_concat is not None:
                    full_grad_concat = torch.concat([full_grad_concat, flat_grad])
                else:
                    full_grad_concat = flat_grad
        print("Tensor dimensions:", full_grad_concat.shape)
        # torch.cuda.empty_cache()
        return full_grad_concat
    
    def domain_reweighting(self, domain_grad_dict, prev_domain_weight):
        domain_score_dict = {}
        for domain_id1, batch1 in domain_grad_dict.items(): # dict
            batch1 = batch1.cpu()
            learnability_scores = batch1 @ batch1.T
            other_domain_sum = torch.zeros_like(batch1)
            for domain_id2, batch2 in domain_grad_dict.items():
                batch2 = batch2.cpu()
                if domain_id2 == domain_id1:
                    other_domain_sum += batch2
            general_impact_score = batch1 @ other_domain_sum.T
            domain_score_dict[domain_id1] = learnability_scores + general_impact_score
            
        # normalization
        sum_score = sum(prev_domain_weight[domain_id] * np.exp(value) for domain_id, value in domain_score_dict.items())
        for domain_id, score in domain_score_dict.items():
            domain_score_dict[domain_id] = (prev_domain_weight[domain_id] * np.exp(score) / sum_score).item()
            
        return domain_score_dict