r"""
Custom Gate
"""
from fmoe.gates.base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np 

__all__ = ['CustomNaiveGate', 'CustomDropGate', 'CustomRandomGate', 'CustomRandomGate_Dense',
            'CustomDTSGate', 'CustomDTSRandomGate', 'CustomDTSGate_softmax', 'CustomDTSRandomGate_softmax',
            'CustomDenseGate', 'CustomHashGate', 'CustomNaiveGate_Balance', 'CustomNaiveGate_Attn']


class CustomNaiveGate(BaseGate):
    r"""
    Naive Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomNaiveGate_Attn(BaseGate):
    r"""
    Naive Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomNaiveGate_Balance(BaseGate):
    r"""
    Naive Gate with Balance loss
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            ) / valid_idx.numel()
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


class CustomHashGate(BaseGate):

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k

    def forward(self, inp, return_all_scores=False):

        if not hasattr(self, 'hash_gate'):
            # generate hash gate
            print('Generate Hash Mapping')
            token_num = inp.shape[0]
            self.register_buffer('hash_gate', torch.rand(token_num, self.tot_expert).to(inp.device))
            print(self.hash_gate.shape)
        else:
            if self.hash_gate.shape[0] != inp.shape[0]:
                if not hasattr(self, 'hash_gate_v2'):
                    print('Generate New Hash Mapping v2')
                    token_num = inp.shape[0]
                    self.register_buffer('hash_gate_v2', torch.rand(token_num, self.tot_expert).to(inp.device))
                    print(self.hash_gate_v2.shape)
                else:
                    if self.hash_gate_v2.shape[0] != inp.shape[0]:
                        if not hasattr(self, 'hash_gate_v3'):
                            print('Generate New Hash Mapping v3')
                            token_num = inp.shape[0]
                            self.register_buffer('hash_gate_v3', torch.rand(token_num, self.tot_expert).to(inp.device))
                            print(self.hash_gate_v3.shape)
                        else:
                            if self.hash_gate_v3.shape[0] != inp.shape[0]:
                                if not hasattr(self, 'hash_gate_v4'):
                                    print('Generate New Hash Mapping v4')
                                    token_num = inp.shape[0]
                                    self.register_buffer('hash_gate_v4', torch.rand(token_num, self.tot_expert).to(inp.device))
                                    print(self.hash_gate_v4.shape)
                                else:
                                    if self.hash_gate_v4.shape[0] != inp.shape[0]:
                                        print('Generate New Hash Mapping v5')
                                        token_num = inp.shape[0]
                                        self.register_buffer('hash_gate_v5', torch.rand(token_num, self.tot_expert).to(inp.device))
                                        print(self.hash_gate_v5.shape)

        if inp.shape[0] == self.hash_gate.shape[0]:
            gate = self.hash_gate
        elif inp.shape[0] == self.hash_gate_v2.shape[0]:
            gate = self.hash_gate_v2
        elif inp.shape[0] == self.hash_gate_v3.shape[0]:
            gate = self.hash_gate_v3
        elif inp.shape[0] == self.hash_gate_v4.shape[0]:
            gate = self.hash_gate_v4
        elif inp.shape[0] == self.hash_gate_v5.shape[0]:
            gate = self.hash_gate_v5
        else:
            assert False

        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_top_k_val = torch.ones_like(gate_top_k_val)
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score




class CustomDropGate(BaseGate):
    r"""
    Dropout Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.training:
            gate = self.dropout(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate) # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomRandomGate(BaseGate):
    r"""
    Random Assign Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        # random gate uniform distribution
        gate = torch.rand_like(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomRandomGate_Dense(BaseGate):
    r"""
    Random Assign Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        # random gate uniform distribution
        gate = torch.ones_like(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score


# Dense to Sparse
class CustomDTSGate(BaseGate):
    r"""
    Dense to Sparse Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
            gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDTSRandomGate(BaseGate):
    r"""
    Dense to Sparse Gate Random Assign
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        gate = torch.rand_like(gate)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
            gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDTSGate_softmax(BaseGate):
    r"""
    Dense to Sparse Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            gate = F.softmax(gate, dim=-1)
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.top_k)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDTSRandomGate_softmax(BaseGate):
    r"""
    Dense to Sparse Gate Random Assign
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

        self.temperature = 1
        self.threshold = 0.001
        self.sum_top_k = 0
        self.forward_n = 0
        self.dynamic_top_k = top_k

    def _sample_gumbel(self, tensor, eps=1e-10):
        U = torch.rand_like(tensor).uniform_()
        return - torch.log(eps - torch.log(U + eps))

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        gate = torch.rand_like(gate)

        if self.training:
            # dts
            gumber_noise = self._sample_gumbel(gate)
            gate_noise = (gate + gumber_noise) / self.temperature
            gate_noise = F.softmax(gate_noise, dim=-1)

            # calculate top-k number 
            enable_gate_number = gate_noise.gt(self.threshold).sum(dim=-1)
            dynamic_top_k = enable_gate_number.float().mean().int().item()
            self.dynamic_top_k = max(self.top_k, dynamic_top_k)

            self.forward_n += 1
            self.sum_top_k += self.dynamic_top_k

            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_noise, k=self.dynamic_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.dynamic_top_k)

        else:
            gate = F.softmax(gate, dim=-1)
            self.dynamic_top_k = self.top_k
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_score = gate_top_k_val.view(-1, self.top_k)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class CustomDenseGate(BaseGate):
    r"""
    Dense Gate
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        repeat_shape = list(gate.shape[:-1])
        repeat_shape.append(1)

        gate_top_k_idx = torch.arange(self.tot_expert).repeat(repeat_shape).to(gate.device)

        gate_top_k_val = gate.view(-1, self.tot_expert)
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score








