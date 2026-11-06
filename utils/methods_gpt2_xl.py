from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils_gpt2_xl import *
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
        if "attn.c_attn.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=1)
            num_cols = w_q.shape[1]
            device = w_q.device
            coeff_q, coeff_k, coeff_v = torch.randint(0,5,(w_q.shape[1],), device=device), torch.randint(0,5,(w_k.shape[1],), device=device), torch.randint(0,5,(w_v.shape[1],), device=device)
            mask_q, mask_k, mask_v = torch.matmul(w_q, coeff_q.float()), torch.matmul(w_k, coeff_k.float()), torch.matmul(w_v, coeff_v.float())
            layer_masks[name] = {"q": mask_q, "k": mask_k, "v": mask_v}
            ratios_q, ratios_k, ratios_v = [], [], []
            ratios_q2, ratios_k2, ratios_v2 = [], [], []
            for i in range(num_cols):
                ratio_q, ratio_k, ratio_v = (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float()
                weight_q, weight_k, weight_v = torch.randint(1, 3, (1,), device=device).float(), torch.randint(1, 3, (1,), device=device).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_qi = mask_q*ratio_q
                mask_ki = mask_k*ratio_k
                mask_vi = mask_v*ratio_v
                w_q[:,i] *= weight_q
                w_k[:,i] *= weight_k
                w_v[:,i] *= weight_v
                w_q[:,i] += mask_qi
                w_k[:,i] += mask_ki
                w_v[:,i] += mask_vi
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
                ratios_q2.append(weight_q)
                ratios_k2.append(weight_k)
                ratios_v2.append(weight_v)
            layer_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            weight_factors[name] = {"q": ratios_q2, "k": ratios_k2, "v": ratios_v2}
            permutation_q = torch.randperm(num_cols)
            permutation_k = torch.randperm(num_cols)
            permutation_v = torch.randperm(num_cols)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[:,permutation_q]
            ob_w_k = w_k[:,permutation_k]
            ob_w_v = w_v[:,permutation_v]
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=1)
            module.data = ob_w
        elif "attn.c_proj.weight" in name:
            w_proj = module.data
            num_cols = w_proj.shape[1]
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[1],), device=device)
            mask = torch.matmul(w_proj, coeff.float())
            layer_masks[name] = mask
            ratios, ratios_2 = [], []
            for i in range(num_cols):
                ratio, ratio2 = (torch.randint(0, 11, (1,), device=device)-5).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_i = mask * ratio
                w_proj[:,i] *= ratio2
                w_proj[:,i] += mask_i
                ratios.append(ratio)
                ratios_2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios_2
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[:,permutation]
            module.data = ob_w_proj
        elif "mlp.c_fc.weight" in name:
            w_fc1 = module.data
            num_cols = w_fc1.shape[1]
            device = w_fc1.device
            coeff = torch.randint(0,5,(w_fc1.shape[1],), device=device)
            mask = torch.matmul(w_fc1, coeff.float())
            layer_masks[name] = mask
            ratios, ratios_2 = [], []
            for i in range(num_cols):
                ratio, ratio2 = (torch.randint(0, 11, (1,), device=device)-5).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_i = mask * ratio
                w_fc1[:,i] *= ratio2
                w_fc1[:,i] += mask_i
                ratios.append(ratio)
                ratios_2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios_2
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[:,permutation]
            module.data = ob_w_fc1
        elif "mlp.c_proj.weight" in name:
            w_fc2 = module.data
            num_cols = w_fc2.shape[1]
            device = w_fc2.device
            coeff = torch.randint(0,5,(w_fc2.shape[1],), device=device)
            mask = torch.matmul(w_fc2, coeff.float())
            layer_masks[name] = mask
            ratios, ratios_2 = [], []
            for i in range(num_cols):
                ratio, ratio2 = (torch.randint(0, 11, (1,), device=device)-5).float(), torch.randint(1, 3, (1,), device=device).float()
                mask_i = mask * ratio
                w_fc2[:,i] *= ratio2
                w_fc2[:,i] += mask_i
                ratios.append(ratio)
                ratios_2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios_2
            permutation = torch.randperm(num_cols)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[:,permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def arrowmatch_recover(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "attn.c_attn.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=1)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=1)
            _, _, restore_wq = col_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = col_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = col_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[1]):
                ratio_q = sqrt(torch.var(restore_wq[:,i]).item()/torch.var(pre_wq[:,i]).item())
                ratio_k = sqrt(torch.var(restore_wk[:,i]).item()/torch.var(pre_wk[:,i]).item())
                ratio_v = sqrt(torch.var(restore_wv[:,i]).item()/torch.var(pre_wv[:,i]).item())
                restore_wq[:,i] /= ratio_q
                restore_wk[:,i] /= ratio_k
                restore_wv[:,i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=1)
        elif "attn.c_proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = col_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[1]):
                ratio_o = sqrt(torch.var(restore_wo[:,i]).item()/torch.var(pre_wo[:,i]).item())
                restore_wo[:,i] /= ratio_o
            module.data = restore_wo
        elif "mlp.c_fc.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = col_restore_perm(pre_fc1, ob_fc1)
            for i in range(ob_fc1.shape[1]):
                ratio_fc1 = sqrt(torch.var(restore_fc1[:,i]).item()/torch.var(pre_fc1[:,i]).item())
                restore_fc1[:,i] /= ratio_fc1
            module.data = restore_fc1
        elif "mlp.c_proj.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = col_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[1]):
                ratio_fc2 = sqrt(torch.var(restore_fc2[:,i]).item()/torch.var(pre_fc2[:,i]).item())
                restore_fc2[:,i] /= ratio_fc2
            module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm
