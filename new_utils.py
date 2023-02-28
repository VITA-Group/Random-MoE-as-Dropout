import copy
import torch
import numpy as np
import torch.nn as nn 
from fmoe.gates.base_gate import BaseGate
from custom_gate import CustomNaiveGate_Attn

import pdb
import torch.nn.functional as F


__all__ = ['set_top_k', 'set_router_mode', 'freeze_part_weight', 'adjust_moe_gate_number',
            'show_dts_gate_number', 'set_temperature', 'set_threshold', 
            'SWA_Average', 'collect_top_k', 'THOR_Model']


def set_top_k(model, num=2):
    for name, m in model.named_modules():
        if hasattr(m, 'top_k') and hasattr(m, 'gate'):
            if isinstance(m.gate, BaseGate) and not isinstance(m.gate, CustomNaiveGate_Attn):
                m.top_k = num
                m.gate.top_k = num
                print('Layer name: {}, Top-K = {}, {}'.format(name, m.top_k, m.gate.top_k))

def collect_top_k(model):
    top_k = None
    for name, m in model.named_modules():
        if hasattr(m, 'top_k') and hasattr(m, 'gate'):
            if isinstance(m.gate, BaseGate) and not isinstance(m.gate, CustomNaiveGate_Attn):
                top_k = m.gate.top_k
                break 
    return top_k

def set_router_mode(model, args, flag=True):
    # for name, m in model.named_modules():
    #     if isinstance(m, BaseGate):
    #         m.dense_moe_flag = flag 
    #         print('Layer name: {}, Average MoE = {}'.format(name, m.dense_moe_flag))
    print('** Using Score-Based Average for Dense Inference')

    current_gate = 0
    for name, m in model.named_modules():
        if hasattr(m, 'top_k') and hasattr(m, 'gate'):
            if isinstance(m.gate, BaseGate) and not isinstance(m.gate, CustomNaiveGate_Attn):
                if flag:
                    m.top_k = args.moe_num_expert
                    m.gate.top_k = args.moe_num_expert
                else:
                    m.top_k = args.moe_top_k
                    m.gate.top_k = args.moe_top_k
                current_gate = m.top_k
                print('Set {}, Top-K = {} {}'.format(name, m.top_k, m.gate.top_k))
    return current_gate

def kl_loss_sym(logits1, logits2):

    kl_loss = nn.KLDivLoss(reduction="none")
    
    loss = kl_loss(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1)) + kl_loss(F.log_softmax(logits2, dim=1), F.softmax(logits1, dim=1))

    return loss.mean(-1)

def freeze_part_weight(model, args):
    if args.freeze_gate:
        print('* Freeze Router')
        for name, p in model.named_parameters():
            if 'gate.gate' in name:
                p.requires_grad = False

    if args.freeze_main_network:
        print('* Freeze All')
        for name, p in model.named_parameters():
            if '.experts.' in name:
                p.requires_grad = False

    if args.freeze_main_network_all:
        print('* Freeze Attention')
        for name, p in model.named_parameters():
            if 'word_emb.emb_layers' in name: continue
            if 'crit.out_layers' in name: continue 
            if 'layers.' in name:
                if not 'gate.gate' in name:
                    p.requires_grad = False

    for name, p in model.named_parameters():
        if p.requires_grad:
            print('* Trainable Parameters {}, shape = {}'.format(name, p.shape))
        else:
            print('* Freeze Parameters {}, shape = {}'.format(name, p.shape))

def calculate_gate_number(steps, args, overall_steps, min_experts, max_experts):
    if args.dynamic_moe_mode == 'linear_increase':
        number_experts = max_experts - min_experts
        gate_num = round(number_experts * steps / overall_steps) + min_experts
    elif args.dynamic_moe_mode == 'linear_decrease':
        number_experts = min_experts - max_experts
        gate_num = round(number_experts * steps / overall_steps) + max_experts
    elif args.dynamic_moe_mode == 'cosine_decrease':
        number_experts = max_experts - min_experts
        cosine_value = np.cos(np.pi * steps / (2 * overall_steps))
        gate_num = round(number_experts * cosine_value) + min_experts
    elif args.dynamic_moe_mode == 'cosine_increase':
        number_experts = min_experts - max_experts
        cosine_value = np.cos(np.pi * steps / (2 * overall_steps))
        gate_num = round(number_experts * cosine_value) + max_experts
    elif args.dynamic_moe_mode == 'exp_increase':
        number_experts = min_experts - max_experts
        current_steps = steps // (overall_steps // 300)
        cosine_value = 0.99 ** current_steps
        gate_num = round(number_experts * cosine_value) + max_experts
    elif args.dynamic_moe_mode == 'multi_step_increase':
        custom_gate_number = [1,2,4,8,16]
        length = len(custom_gate_number)
        gate_num_index = int(length * steps / overall_steps)
        gate_num = custom_gate_number[gate_num_index]
    elif args.dynamic_moe_mode == 'multi_step_decrease':
        custom_gate_number = [16,8,4,2,1]
        length = len(custom_gate_number)
        gate_num_index = int(length * steps / overall_steps)
        gate_num = custom_gate_number[gate_num_index]

    gate_num = np.clip(gate_num, min_experts, max_experts)

    return gate_num

def adjust_moe_gate_number(model, steps, args, current_gate):
    new_gate_num = calculate_gate_number(steps, args, args.dynamic_overall_steps, args.moe_top_k_min, args.moe_top_k_max)
    if new_gate_num != current_gate:
        print('* Set New Top-k = {}'.format(new_gate_num))
        set_top_k(model, new_gate_num)
        current_gate = new_gate_num
    return current_gate


## Dense to Sparse
def show_dts_gate_number(model):
    for name, m in model.named_modules():
        if isinstance(m, BaseGate):
            mean_experts = m.sum_top_k / m.forward_n
            layer_temp = m.temperature
            layer_threshold = m.threshold
            print('* Mean-Experts = {:.0f}, Temperature = {:.4f}, Threshold = {:.4f}'.format(mean_experts, layer_temp, layer_threshold))

def set_temperature(model, iterations, all_iteration, max_temp, min_temp):
    temp = max_temp + iterations * (min_temp - max_temp) / all_iteration
    for name, m in model.named_modules():
        if isinstance(m, BaseGate):
            m.temperature = temp

def set_threshold(model, args):
    if args.gate_name == 'CustomDTSGate':
        print('* Set threshold for DTS Gate')
        for name, m in model.named_modules():
            if isinstance(m, BaseGate):
                m.threshold = args.threshold



## Weight Average
class SWA_Average(nn.Module):
    def __init__(self, model, t_start, t_end, device):
        super(SWA_Average, self).__init__()
        self.device = device
        self.average_model = copy.deepcopy(model) 
        self.register_buffer('n_average', torch.tensor(0, dtype=torch.long, device=self.device))
        self.t_start = t_start
        self.t_end = t_end 
    
    def forward(self, data, target, *mems):
        return self.average_model(data, target, *mems)
    
    def avg_fn(self, averaged_model_parameter, model_parameter, num_averaged):
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
            num_averaged + 1
        )

    def update_parameters(self, current_model, step):
        if step >= self.t_start and step <= self.t_end:
            print('Update parameters with step {}, current_n_average = {}'.format(step, self.n_average))
            for p_swa, p_model in zip(self.average_model.parameters(), current_model.parameters()):
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model.detach(), self.n_average))
            self.n_average +=1 

class THOR_Model(nn.Module):
    def __init__(self, basic_model, kl_alpha):
        super(THOR_Model, self).__init__()
        self.module = basic_model
        self.kl_alpha = kl_alpha
    
    def reset_length(self, tgt_len, ext_len, mem_len):
        self.module.reset_length(tgt_len, ext_len, mem_len)

    def forward(self, data, target, *mems):
        if self.training:
            outputs = self.module(data, target, *mems)
            outputs2 = self.module(data, target, *mems)
            loss_kl = kl_loss_sym(outputs[0], outputs2[0])
            new_loss = (outputs[1] + outputs2[1])/2 + self.kl_alpha * loss_kl
            outputs[1] = new_loss
        else:
            outputs = self.module(data, target, *mems)
        return outputs[1:]

