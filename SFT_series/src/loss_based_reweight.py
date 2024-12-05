import json
import ujson
import numpy as np
import copy
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

        
class ReweightTrainer(Trainer):
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)
    
    def loss_based_reweighting(self, 
                               dataloader,
                               model,
                               prev_domain_weight=None):
        torch.random.manual_seed(0)  # set the random seed for torch
        device = next(model.parameters()).device
        print("Device:", device)
        
        domain_loss_dict = {}
        for domain in dataloader.keys():
            domain_loss_dict[domain] = []
            cur_data_loader = dataloader[domain]
            domain_loss_dict[domain] = self.obtain_loss(cur_data_loader, model)
        
        domain_weight_dict = self.domain_reweighting(domain_loss_dict, prev_domain_weight)
        print("domain_weight_dict", domain_weight_dict)
        model.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("Finished")
        return domain_weight_dict
    
       
    def obtain_loss(self, dataloader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,): # put all tensors on the same device
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
        # with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        #     f.write(json.dumps(result, indent=4))
        return result["loss"]
    
    
    def domain_reweighting(self, domain_loss_dict, prev_domain_weight, ref_loss=None): # ref_loss: the loss of the proxy model on the current domain
        domain_score_dict = {}
        for domain_id, loss_value in domain_loss_dict.items(): # keys: domain_name, values: loss
            loss_diff = loss_value - ref_loss[domain_id] if (ref_loss is not None and domain_id in ref_loss) else loss_value
            # domain_score_dict[domain_id] = loss_diff / loss_value # official version
            domain_score_dict[domain_id] = loss_diff
            # if domain_id == "general":
            #     domain_score_dict["other"] = domain_score_dict[domain_id]
            #     del domain_score_dict[domain_id]

        domain_score_dict["other"] = domain_score_dict["general"]
        del domain_score_dict["general"]
        print("domain_score_dict", domain_score_dict) # key: general
        print("prev_domain_weight", prev_domain_weight) # key: other
        # normalization
        sum_score = sum(prev_domain_weight[domain_id] * np.exp(value) for domain_id, value in domain_score_dict.items())
        for domain_id, score in domain_score_dict.items():
            domain_score_dict[domain_id] = (prev_domain_weight[domain_id] * np.exp(score) / sum_score).item()
            
        return domain_score_dict