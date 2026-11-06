from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt

def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            w_q = module.data
            num_rows = w_q.shape[0]
            device = w_q.device
            coeff = torch.randint(0,5,(w_q.shape[0],), device=device)
            mask = torch.matmul(w_q.T, coeff.float())
            layer_masks[name] = mask
            ratios_q = []
            ratios_q2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= ratio2
                mask_qi = mask * ratio
                w_q[i] += mask_qi
                ratios_q.append(ratio)
                ratios_q2.append(ratio2)
            layer_factors[name] = ratios_q
            weight_factors[name] = ratios_q2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_q = w_q[permutation]
            module.data = w_q
        elif "key.weight" in name:
            w_k = module.data
            num_rows = w_k.shape[0]
            device = w_k.device
            coeff = torch.randint(0,5,(w_k.shape[0],), device=device)
            mask = torch.matmul(w_k.T, coeff.float())
            layer_masks[name] = mask
            ratios_k = []
            ratios_k2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_k[i] *= ratio2
                mask_ki = mask * ratio
                w_k[i] += mask_ki
                ratios_k.append(ratio)
                ratios_k2.append(ratio2)
            layer_factors[name] = ratios_k
            weight_factors[name] = ratios_k2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_k = w_k[permutation]
            module.data = w_k
        elif "value.weight" in name:
            w_v = module.data
            num_rows = w_v.shape[0]
            device = w_v.device
            coeff = torch.randint(0,5,(w_v.shape[0],), device=device)
            mask = torch.matmul(w_v.T, coeff.float())
            layer_masks[name] = mask
            ratios_v = []
            ratios_v2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_v[i] *= ratio2
                mask_vi = mask * ratio
                w_v[i] += mask_vi
                ratios_v.append(ratio)
                ratios_v2.append(ratio2)
            layer_factors[name] = ratios_v
            weight_factors[name] = ratios_v2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_v = w_v[permutation]
            module.data = w_v
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            w_proj = module.data
            num_rows = w_proj.shape[0]
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(num_rows):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            permutation = torch.randperm(num_rows)
            layer_permutations[name] = permutation
            w_proj = w_proj[permutation]
            module.data = w_proj
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def arrowmatch_recover(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "query.weight" in name:
            ob_wq = module.data
            pre_wq = pre_model.state_dict()[name].data
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            for i in range(ob_wq.shape[0]):
                ratio_q = sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item())
                restore_wq[i] /= ratio_q
            module.data = restore_wq
        elif "key.weight" in name:
            ob_wk = module.data
            pre_wk = pre_model.state_dict()[name].data
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            for i in range(ob_wk.shape[0]):
                ratio_k = sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item())
                restore_wk[i] /= ratio_k
            module.data = restore_wk
        elif "value.weight" in name:
            ob_wv = module.data
            pre_wv = pre_model.state_dict()[name].data
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wv.shape[0]):
                ratio_v = sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item())
                restore_wv[i] /= ratio_v
            module.data = restore_wv
        elif "output.dense.weight" in name or "intermediate.dense.weight" in name:
            ob_proj = module.data
            pre_proj = pre_model.state_dict()[name].data
            _, _, restore_proj = row_restore_perm(pre_proj, ob_proj)
            for i in range(ob_proj.shape[0]):
                ratio = sqrt(torch.var(restore_proj[i]).item()/torch.var(pre_proj[i]).item())
                restore_proj[i] /= ratio
            module.data = restore_proj
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm