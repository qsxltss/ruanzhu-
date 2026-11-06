import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdb import set_trace as st
import torch
import numpy as np
import random
from utils.utils_vit import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from matplotlib import pyplot as plt


def ob_arrowcloak(model):
    set_seed()
    layer_permutations = {}
    layer_masks = {}
    layer_factors = {}
    weight_factors = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            w_q, w_k, w_v = module.data.chunk(3, dim=0)
            device = w_q.device
            coeff_q, coeff_k, coeff_v = torch.randint(0,5,(w_q.shape[0],), device=device), torch.randint(0,5,(w_k.shape[0],), device=device), torch.randint(0,5,(w_v.shape[0],), device=device)
            mask_q, mask_k, mask_v = torch.matmul(w_q.T, coeff_q.float()), torch.matmul(w_k.T, coeff_k.float()), torch.matmul(w_v.T, coeff_v.float())
            layer_masks[name] = {"q": mask_q, "k": mask_k, "v": mask_v}
            ratios_q, ratios_k, ratios_v = [], [], []
            ratios_q2, ratios_k2, ratios_v2 = [], [], []
            for i in range(w_q.shape[0]):
                ratio_q, ratio_k, ratio_v = (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float(), (torch.randint(0, 11, (1,), device=device)-5).float()
                weight_q, weight_k, weight_v = (torch.randint(1, 3, (1,), device=device)).float(), (torch.randint(1, 3, (1,), device=device)).float(), (torch.randint(1, 3, (1,), device=device)).float()
                w_q[i] *= weight_q
                w_k[i] *= weight_k
                w_v[i] *= weight_v
                mask_qi, mask_ki, mask_vi = mask_q * ratio_q, mask_k * ratio_k, mask_v * ratio_v
                w_q[i] += mask_qi
                w_k[i] += mask_ki
                w_v[i] += mask_vi
                ratios_q.append(ratio_q)
                ratios_k.append(ratio_k)
                ratios_v.append(ratio_v)
                ratios_q2.append(weight_q)
                ratios_k2.append(weight_k)
                ratios_v2.append(weight_v)
            layer_factors[name] = {"q": ratios_q, "k": ratios_k, "v": ratios_v}
            weight_factors[name] = {"q": ratios_q2, "k": ratios_k2, "v": ratios_v2}
            rows = w_q.shape[0]
            permutation_q = torch.randperm(rows)
            permutation_k = torch.randperm(rows)
            permutation_v = torch.randperm(rows)
            layer_permutations[name] = {"q": permutation_q, "k": permutation_k, "v": permutation_v}
            ob_w_q = w_q[permutation_q]
            ob_w_k = w_k[permutation_k]
            ob_w_v = w_v[permutation_v] 
            ob_w = torch.cat([ob_w_q, ob_w_k, ob_w_v], dim=0)
            module.data = ob_w
        if "attn.proj.weight" in name:
            w_proj = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_proj.shape[0],), device=device)
            mask = torch.matmul(w_proj.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_proj.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_proj[i] *= ratio2
                mask_i = mask * ratio
                w_proj[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_proj.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_proj = w_proj[permutation]
            module.data = ob_w_proj
        if "mlp.fc1.weight" in name:
            w_fc1 = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_fc1.shape[0],), device=device)
            mask = torch.matmul(w_fc1.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_fc1.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_fc1[i] *= ratio2
                mask_i = mask * ratio 
                w_fc1[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_fc1.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_fc1 = w_fc1[permutation]
            module.data = ob_w_fc1
        if "mlp.fc2.weight" in name:
            w_fc2 = module.data
            device = w_proj.device
            coeff = torch.randint(0,5,(w_fc2.shape[0],), device=device)
            mask = torch.matmul(w_fc2.T, coeff.float())
            layer_masks[name] = mask
            ratios = []
            ratios2 = []
            for i in range(w_fc2.shape[0]):
                ratio = (torch.randint(0, 11, (1,), device=device)-5).float()
                ratio2 = (torch.randint(1, 3, (1,), device=device)).float()
                w_fc2[i] *= ratio2
                mask_i = mask * ratio 
                w_fc2[i] += mask_i
                ratios.append(ratio)
                ratios2.append(ratio2)
            layer_factors[name] = ratios
            weight_factors[name] = ratios2
            rows = w_fc2.shape[0]
            permutation = torch.randperm(rows)
            layer_permutations[name] = permutation
            ob_w_fc2 = w_fc2[permutation]
            module.data = ob_w_fc2
    return model, layer_permutations, layer_masks, layer_factors, weight_factors

def arrowmatch_recover(model, pre_model):
    set_seed()
    restore_perm = {}
    for name, module in model.named_parameters():
        if "qkv.weight" in name:
            ob_wq, ob_wk, ob_wv = module.data.chunk(3, dim=0)
            pre_wq, pre_wk, pre_wv = pre_model.state_dict()[name].data.chunk(3, dim=0)
            ## restore_wq是恢复了permutation的wq
            _, _, restore_wq = row_restore_perm(pre_wq, ob_wq)
            _, _, restore_wk = row_restore_perm(pre_wk, ob_wk)
            _, _, restore_wv = row_restore_perm(pre_wv, ob_wv)
            for i in range(ob_wq.shape[0]):
                ratio_q = sqrt(torch.var(restore_wq[i]).item()/torch.var(pre_wq[i]).item())
                ratio_k = sqrt(torch.var(restore_wk[i]).item()/torch.var(pre_wk[i]).item())
                ratio_v = sqrt(torch.var(restore_wv[i]).item()/torch.var(pre_wv[i]).item())
                restore_wq[i] /= ratio_q
                restore_wk[i] /= ratio_k
                restore_wv[i] /= ratio_v
            module.data = torch.cat([restore_wq, restore_wk, restore_wv], dim=0)
        if "attn.proj.weight" in name:
            ob_wo = module.data
            pre_wo = pre_model.state_dict()[name].data
            _, _, restore_wo = row_restore_perm(pre_wo, ob_wo)
            for i in range(ob_wo.shape[0]):
                ratio_o = sqrt(torch.var(restore_wo[i]).item()/torch.var(pre_wo[i]).item())
                restore_wo[i] /= ratio_o
            module.data = restore_wo
        if "mlp.fc1.weight" in name:
            ob_fc1 = module.data
            pre_fc1 = pre_model.state_dict()[name].data
            _, _, restore_fc1 = row_restore_perm(pre_fc1, ob_fc1)   
            for i in range(ob_fc1.shape[0]):
                ratio_fc1 = sqrt(torch.var(restore_fc1[i]).item()/torch.var(pre_fc1[i]).item())
                restore_fc1[i] /= ratio_fc1
            module.data = restore_fc1
        if "mlp.fc2.weight" in name:
            ob_fc2 = module.data
            pre_fc2 = pre_model.state_dict()[name].data
            _, _, restore_fc2 = row_restore_perm(pre_fc2, ob_fc2)
            for i in range(ob_fc2.shape[0]):
                ratio_fc2 = sqrt(torch.var(restore_fc2[i]).item()/torch.var(pre_fc2[i]).item())
                restore_fc2[i] /= ratio_fc2
            module.data = restore_fc2
        else:
            module.data = pre_model.state_dict()[name].data
    return model, restore_perm

def recover_finetune(model, trainloader, evalloader, num_classes, save_path, device, size=224, epochs=3, lr=5e-6):
    set_seed()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()    
    
    model = model.to(device)

    training_losses = []
    eval_losses = []
    for epoch in range(epochs):
        model.train()
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        epoch_progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Training")
        tot_train_loss = 0  
        for batch_idx, (images, labels) in epoch_progress_bar:
            images, labels = images.to(device), labels.to(device)
            if images.size(-1) != size:
                images = torch.nn.functional.interpolate(images, size=(size, size))
            optimizer.zero_grad()  
            outputs = model(images)
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            tot_train_loss += loss.item()
            epoch_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = tot_train_loss / len(trainloader)
        training_losses.append(avg_train_loss) 
           
        scheduler.step()  
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}], Current LR: {current_lr:.6e}")
        
        ## 计算验证损失
        model.eval()
        tot_eval_loss = 0
        with torch.no_grad():
            for images, labels in evalloader:
                images, labels = images.to(device), labels.to(device)
                if images.size(-1) != size:
                    images = torch.nn.functional.interpolate(images, size=(size, size))
                outputs = model(images)
                loss = criterion(outputs, labels)
                tot_eval_loss += loss.item()

        avg_eval_loss = tot_eval_loss / len(evalloader)
        eval_losses.append(avg_eval_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Eval Loss: {avg_eval_loss:.6f}")
    
    save_model_weights(model, save_path)
    print(f"Model weights saved to {save_path}") 
    return model

