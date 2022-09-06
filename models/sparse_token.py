"""
Sparse Token Transformer with Attetion Back Tracking
Heejun Lee
2022
"""

import copy
import datetime
import math
import os
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions, MaskedLMOutput,
    MultipleChoiceModelOutput, NextSentencePredictorOutput,
    QuestionAnsweringModelOutput, SequenceClassifierOutput,
    TokenClassifierOutput)
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.bert.modeling_bert import \
    BertModel as OriginalBertModel
from transformers.utils import logging
from transformers.models.vit.modeling_vit import ViTEmbeddings

from utils.sparse_flops_calculation import flops_sparse_approx_bert_model
from utils.sparse_flops_calculation import ModelConfig as FlopsConfig

logger = logging.get_logger(__name__)

EPS = 1e-7
USE_LTP_ON_CONCRETE = False

#region Timer

__timer_cum = {}
__timer_start = {}
__timer_enable = True

def timer_enable(v):
    global __timer_enable
    __timer_enable = v

def timer_start(name):
    global __timer_start, __timer_enable
    if not __timer_enable: return

    __timer_start[name] = datetime.datetime.now()

def timer_end(name):
    global __timer_start, __timer_cum, __timer_enable
    if not __timer_enable: return

    if not name in __timer_cum: __timer_cum[name] = datetime.timedelta(0)
    __timer_cum[name] += datetime.datetime.now() - __timer_start[name]

def timer_reset():
    global __timer_cum, __timer_enable
    if not __timer_enable: return
    __timer_cum = {}

def timer_report():
    global __timer_cum, __timer_enable
    if not __timer_enable: return
    data = __timer_cum
    name_max = max(map(lambda x: len(x), data.keys()))
    total_time = datetime.timedelta(0)
    for k in data.keys(): total_time += data[k]
    for k in sorted(data.keys()):
        print(f'{k.ljust(name_max+1)}: {data[k]} ({(data[k]/total_time)*100:2.2f}%)')

#endregion Timer

#region Benchmark

__benchmark = {}

def benchmark_cum(name, value):
    global __benchmark
    if not name in __benchmark:
        __benchmark[name] = (0, 0)
    count, v = __benchmark[name]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    __benchmark[name] = (count + 1, v + value)

def benchmark_get_average(name):
    global __benchmark
    if name in __benchmark:
        c, v = __benchmark[name]
        if isinstance(v, float) or isinstance(v, int):
            return v/c
        elif isinstance(v, torch.Tensor):
            return (v/c).item()
        else:
            raise Exception('unknown benchmark dtype')
    return 0

def benchmark_report():
    global __benchmark
    for k in __benchmark.keys():
        c, v = __benchmark[k]
        print(f"{k}: {__benchmark[k]}, avg: {v/c}")

def benchmark_reset():
    global __benchmark
    __benchmark = {}

BENCHMARK_LTP_OCCUPY = True
def benchmark_ltp_occupy(v):
    global BENCHMARK_LTP_OCCUPY
    BENCHMARK_LTP_OCCUPY = v

BENCHMARK_CONCRETE_OCCUPY = True
def benchmark_concrete_occupy(v):
    global BENCHMARK_CONCRETE_OCCUPY
    BENCHMARK_CONCRETE_OCCUPY = v

BENCHMARK_SPARSE_APPROX_FLOPS = False
def benchmark_sparse_approx_flops(v):
    global BENCHMARK_SPARSE_APPROX_FLOPS
    BENCHMARK_SPARSE_APPROX_FLOPS = v

#endregion

#region Model Definition

#very bad bad global flags. THIS IS BAD EVILs
__mask_acc_indices = True
def set_update_input_mask_accumulate_indices(value):
    global __mask_acc_indices
    __mask_acc_indices = value

def update_input_mask_from_previous_attention(
    attention_mask, # mask may None (vit)
    previous_attention, 
    output_token_indices, 
    output_token_impact, 
    head_reduce_method = 'avg',
    token_reduce_method = 'avg',
    apply_token_impact = True,
    k=0.5,
    k_estimate = False,
    accumulate_indices = None,
):
    global __mask_acc_indices
    if accumulate_indices is None:
        accumulate_indices = __mask_acc_indices
    
    timer_start('update_mask')
    # attention_mask_shape = attention_mask.shape
    # NBATCH = attention_mask_shape[0]
    # TLEN = attention_mask_shape[-1]
    NBATCH, NHEAD, TLEN, _T = previous_attention.shape
    assert TLEN == _T
    dtype = previous_attention.dtype
    device = previous_attention.device
    if isinstance(k, str):
        if k == 'dynamic':
            k_estimate = True
        elif isinstance(k, str) and k.startswith('dynamic'):
            k_estimate = True
            _, head_method, token_method, apply_token_impact = k.split(':')
            head_reduce_method = head_method
            token_reduce_method = token_method
            apply_token_impact = apply_token_impact.lower() == 'true'
        else: raise Exception(f'unknown method {k}')
    else:
        if k < 1.0:
            kxx = int(math.ceil(k*TLEN))
        else:
            kxx = k
    
    timer_start('update_mask.reduce')
    if head_reduce_method == 'avg':
        att = torch.mean(previous_attention, dim=1)  #reduce head
    elif head_reduce_method == 'max':
        att = torch.max(previous_attention, dim=1)[0]
    else: raise Exception()

    att = torch.gather(att, 1, output_token_indices.unsqueeze(-1).expand(NBATCH, output_token_indices.shape[1], TLEN))
    if apply_token_impact: 
        #todo fix this..
        att = att * output_token_impact.unsqueeze(-1) * 0.1 + att * 0.9
    
    if token_reduce_method == 'avg':
        att = torch.mean(att, dim=1)                 #reduce token, column mean
    elif token_reduce_method == 'max':
        att = torch.max(att, dim=1)[0]
    else: raise Exception()
    timer_end('update_mask.reduce')
    #att(N, TLEN)
    
    timer_start('update_mask.topk')
    if k_estimate:
        att_max = torch.max(att, dim=1, keepdim=True)[0]
        #att_mean = torch.sum(att, dim=1, keepdim=True) / (torch.sum(attention_mask + 10001, dim=1, keepdim=True) + 1e-8)
        #assert torch.min(attention_mask) == -10000
        #att_max = att_max * 0.1 + att_mean * 0.9
        # print(torch.min(att_max), torch.max(att_max), torch.min(att), torch.max(att))
        # input()
        est_k = min(math.ceil(TLEN*0.95), max(math.ceil(TLEN*0.2), math.ceil(torch.max(torch.sum((att > (att_max * 0.00625)) * 1.0, dim=1)).item())))
        benchmark_cum('est_k', est_k / TLEN)
        kxx = est_k
    _, input_indices = torch.topk(att, kxx, dim=1)
    if accumulate_indices:
        input_indices = torch.cat([input_indices, output_token_indices], dim=1)
        input_indices = torch.unique(input_indices, dim=1)
    benchmark_cum('mask_occupy', input_indices.shape[1]/TLEN)
    input_impacts = att.gather(1, input_indices)
    input_impacts = input_impacts / (torch.sum(input_impacts, dim=1, keepdim=True) + 1e-8)
    #input_impacts(N, K), input_indices(N, K)
    
    input_mask = None
    input_mask = torch.zeros(NBATCH, TLEN, device=device, dtype=dtype)\
        .scatter_(1, input_indices, 1.0)
    timer_end('update_mask.topk')
    #input_mask (N, TLEN)
    timer_end('update_mask')
    return input_mask, input_indices, input_impacts

class SparseChannelLinear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        self.channel_indices= None #(N, K), K<=C
        self.force_dense = False

        self.concrete_score = None
        self.concrete_mask = None   #for concrete dropout
        self.concrete_mask_hard = None
        self.concrete_mask_mm = None   #for concrete dropout
        self.concrete_mask_mm_hard = None
        self.concrete_hard_threshold = None
        self.concrete_debug = None
        self.concrete_print = False
        self.retain_prob = 1.0

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.channel_indices is None or self.force_dense:
            timer_start('sparselinear.dense.linear')
            if self.concrete_print:
                print('value', input[0, :, 0].view(-1))
                print('value mask', self.concrete_mask_mm[0].view(-1))
            if self.concrete_mask_mm is not None:
                if self.concrete_mask_mm_hard is not None:
                    input = input * self.concrete_mask_mm_hard
                else:
                    input = input * self.concrete_mask_mm
            x = F.linear(input, self.weight, self.bias)
            timer_end('sparselinear.dense.linear')
        else:
            timer_start('sparselinear')
            input_shape = input.shape
            if len(input_shape) != 3: print(input_shape)
            assert len(input_shape) == 3
            N, C, H = input_shape
            #input (N, C, H)
            #weight (OUT, H)
            #bias (OUT)
            channel_indices_unsqueeze = self.channel_indices#.unsqueeze(-1)
            timer_start('sparselinear.gather')
            sparse_input = torch.gather(
                input, dim=1, index=channel_indices_unsqueeze.expand(-1,-1,self.in_features))
            timer_end('sparselinear.gather')
            #sparse_input (N, T, H)
            timer_start('sparselinear.linear')
            x = F.linear(sparse_input, self.weight, self.bias)
            timer_end('sparselinear.linear')

            timer_start('sparselinear.zeros')
            x_zeros = torch.zeros(N, C, self.out_features, 
                dtype=x.dtype, device=x.device)
            timer_end('sparselinear.zeros')
            timer_start('sparselinear.scatter_')
            x = x_zeros.scatter_(dim=1, index=channel_indices_unsqueeze.expand(-1,-1,self.out_features), src=x) 
            timer_end('sparselinear.scatter_')
            timer_end('sparselinear')
            #return F.linear(input, self.weight, self.bias)
        
        if not self.concrete_mask is None:
            mask = self.concrete_mask
            if self.concrete_mask_hard is not None:
                # mask = (mask > self.concrete_hard_threshold) * 1.0
                # self.concrete_mask_hard = mask
                #assert self.concrete_mask_hard is not None
                mask = self.concrete_mask_hard
            #print(mask[0])
            x = x * mask
            #x = x / (self.retain_prob + EPS) # retain prob makes thing worse and worse!!!
            #print(x.shape, self.concrete_mask.shape)
        
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, arch='bert'):
        super().__init__()
        self.arch = arch
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if arch == 'bert':
            self.query = SparseChannelLinear(config.hidden_size, self.all_head_size)
            self.key = SparseChannelLinear(config.hidden_size, self.all_head_size)
            self.value = SparseChannelLinear(config.hidden_size, self.all_head_size)
        elif arch == 'vit':
            self.query = SparseChannelLinear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
            self.key = SparseChannelLinear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
            self.value = SparseChannelLinear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        else: raise Exception()

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise Exception('removed')

        self.is_decoder = config.is_decoder
        if self.is_decoder: raise Exception()

        self.print = False
        self.reset_input_mask()
        self.attention_masking_timing = 'before_softmax'
        self.output_masking = False
        self.backup_last_inputs = True

        #self.query.force_dense = self.key.force_dense = self.value.force_dense = False
    
    def reset_input_mask(self):
        self.concrete_input_mask = None
        self.input_mask = None
        self.input_indices = None
        self.input_impacts = None
        self.output_mask = None
        self.output_indices = None
        self.query.channel_indices = None
        self.query.concrete_mask = None
        self.query.concrete_mask_hard = None
        self.key.channel_indices = None
        self.key.concrete_mask = None
        self.key.concrete_mask_hard = None
        self.value.channel_indices = None
        self.value.concrete_mask = None
        self.value.concrete_mask_hard = None
    
    def update_input_mask_from_previous_attention(self, output_token_mask, output_token_indices, output_token_impact, k):
        input_mask, input_indices, input_impacts = update_input_mask_from_previous_attention(
            self.last_attention_mask, 
            self.last_attention_probs, 
            output_token_indices, 
            output_token_impact, 
            k=k,
        )
        self.input_mask = input_mask
        self.input_indices = input_indices
        self.input_impacts = input_impacts
        self.output_mask = output_token_mask
        self.output_indices = output_token_indices
        
        input_indices_un = input_indices.unsqueeze(-1)
        output_indices_un = output_token_indices.unsqueeze(-1)
        self.query.channel_indices = output_indices_un
        self.key.channel_indices = input_indices_un
        self.value.channel_indices = input_indices_un

        return input_mask, input_indices, input_impacts

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        if self.is_decoder: raise Exception()
        if not self.attention_masking_timing in ['after_softmax', 'before_softmax']: raise Exception()
        self.last_hidden_states = hidden_states
        timer_start('bert.attention.qkv')
        mixed_query_layer = self.query(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None
        if (is_cross_attention and past_key_value is not None) or is_cross_attention or (past_key_value is not None):
            raise Exception()
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.backup_last_inputs:
            self.last_query = query_layer
            self.last_key = key_layer
            self.last_value = value_layer
        timer_end('bert.attention.qkv')

        timer_start('bert.attention.scores.matmul')
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        timer_end('bert.attention.scores.matmul')
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise Exception()

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.input_mask is not None and self.attention_masking_timing == 'before_softmax':
            if self.print: print(f'apply input mask, before softmax. input_mask:{self.input_mask.shape}, attention_scores:{attention_scores.shape}')
            attention_scores = torch.clamp_min(
                attention_scores + (1.0 - self.input_mask.view(self.input_mask.shape[0], 1, 1, self.input_mask.shape[-1])) * -10000,
                -10000)
        
        if self.backup_last_inputs:
            self.last_attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # if self.input_mask is not None and self.attention_masking_timing == 'before_softmax':
        #     masked_last_probs = self.last_attention_probs * self.input_mask.view(self.input_mask.shape[0], 1, 1, self.input_mask.shape[-1])
        #     masked_last_probs_max = torch.max(masked_last_probs, dim=-1, keepdim=True)[0]
        #     probs_max = torch.max(attention_probs, dim=-1, keepdim=True)[0]
        #     attention_probs = attention_probs * masked_last_probs_max / (probs_max + 1e-8)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        timer_start('bert.attention.probs.dropout')
        attention_probs = self.dropout(attention_probs)
        timer_end('bert.attention.probs.dropout')

        if self.concrete_input_mask is not None:
            N, H, T, _ = attention_probs.shape
            attention_probs = attention_probs * self.concrete_input_mask.view(N, 1, 1, T)
            attention_probs = attention_probs / (torch.sum(attention_probs, dim=-1, keepdim=True) + EPS)

        # Mask heads if we want to
        if head_mask is not None:
            raise "not supported"
            attention_probs = attention_probs * head_mask

        if self.input_mask is not None and self.attention_masking_timing == 'after_softmax':
            raise 'not supported'
            if self.print: print(f'apply input mask, after softmax. input_mask:{self.input_mask.shape}, attention_probs:{attention_probs.shape}')
            attention_probs = attention_probs * self.input_mask.view(self.input_mask.shape[0], 1, 1, self.input_mask.shape[-1])
        
        if self.backup_last_inputs:
            if self.print: print('SelfAttention.forward: last_attention_probs backuped')
            self.last_attention_probs = attention_probs
            self.last_attention_mask = attention_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.output_mask is not None and self.output_masking:
            output_mask = self.output_mask.unsqueeze(-1)
            if self.print: print(f'apply output mask. mask:{output_mask.shape} context:{context_layer.shape}')
            context_layer = context_layer * output_mask

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config, arch='bert'):
        super().__init__()
        self.arch = arch
        self.dense = SparseChannelLinear(config.hidden_size, config.hidden_size)
        if arch == 'bert':
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif arch == 'vit':
            pass
        else: raise Exception()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.arch == 'bert':
            hidden_states = hidden_states + input_tensor
            hidden_states = self.LayerNorm(hidden_states)
        elif self.arch == 'vit':
            pass
        else:
            raise Exception()
        # skip
        # if self.dense.concrete_mask is not None:
        #     if self.dense.concrete_mask_hard is not None:
        #         hidden_states = hidden_states * self.dense.concrete_mask_hard
        #     else:
        #         hidden_states = hidden_states * self.dense.concrete_mask
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, arch='bert'):
        super().__init__()
        self.arch = arch
        if arch == 'bert':
            self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type, arch=arch)
        elif arch == 'vit':
            self.attention = BertSelfAttention(config, position_embedding_type=position_embedding_type, arch=arch)
        else: raise Exception()
        self.output = BertSelfOutput(config, arch=arch)
        self.pruned_heads = set()

    def get_attention(self):
        if self.arch == 'bert':
            return self.self
        elif self.arch == 'vit':
            return self.attention
        else: raise Exception()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        attention = self.get_attention()

        heads, index = find_pruneable_heads_and_indices(
            heads, attention.num_attention_heads, attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        attention.query = prune_linear_layer(attention.query, index)
        attention.key = prune_linear_layer(attention.key, index)
        attention.value = prune_linear_layer(attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        attention.num_attention_heads = attention.num_attention_heads - len(heads)
        attention.all_head_size = attention.attention_head_size * attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        #print('att_preself', hidden_states[0,:,0].view(-1))
        attention = self.get_attention()
        self_outputs = attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        timer_start('bert.attention.output')
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        timer_end('bert.attention.output')
        #print('att_final', outputs[0][0,:,0].view(-1))
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SparseChannelLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        timer_start('bert.intermediate')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        timer_end('bert.intermediate')
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config, arch='bert'):
        super().__init__()
        self.arch = arch
        self.dense = SparseChannelLinear(config.intermediate_size, config.hidden_size)
        if arch == 'bert':
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif arch == 'vit':
            pass
        else: raise Exception()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        timer_start('bert.output')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if self.arch == 'bert':
            hidden_states = self.LayerNorm(hidden_states)
        elif self.arch == 'vit':
            pass
        else:
            raise Exception()
        # skip
        # if self.dense.concrete_mask is not None:
        #     if self.dense.concrete_mask_hard is not None:
        #         hidden_states = hidden_states * self.dense.concrete_mask_hard
        #     else:
        #         hidden_states = hidden_states * self.dense.concrete_mask
        timer_end('bert.output')
        return hidden_states

class LTPPruneToken(nn.Module):
    def __init__(self):
        super().__init__()

        self.soft_pruning = True
        self.threshold = None # nn.Parameter(torch.randn((1,), dtype=torch.float32))
        self.last_mask = None
        self.new_attention_mask = None
        self.temperature = 5e-4
    
    def init_threshold(self, l, L):
        self.threshold = nn.Parameter(torch.tensor([0.01 * l / L], dtype=torch.float32))

    def forward(self, x, attention_score, attention_mask):
        # x: (N, T, H)
        # attention_score: (N, HEAD, T, T)
        N, T0, H = x.shape
        _N, HEAD, T1, T2 = attention_score.shape
        assert T1 == T2
        assert T0 == T1
        T = T1
        assert N == _N

        if self.soft_pruning:
            #score (N, T)
            score = torch.mean(torch.mean(attention_score, dim=1), dim=1)
            self.last_mask = torch.sigmoid((score - self.threshold) / self.temperature)
        else:
            score = torch.mean(torch.mean(attention_score, dim=1), dim=1)
            self.last_mask = (score > self.threshold) * 1.0
            # this is replace the attention mask for next layer. so equivalent to drop the token.
            # have to update attention mask when hard pruning, according to LTP implementation.
            new_attention_mask = (1-self.last_mask) * (-10000)
            attention_mask = new_attention_mask.view(*attention_mask.shape)
        self.last_mask = self.last_mask.unsqueeze(-1) # masking layer output
        if BENCHMARK_LTP_OCCUPY: benchmark_cum("ltp_occupy", self.last_mask.mean())
        self.new_attention_mask = attention_mask
        
        return x * self.last_mask

class BertLayer(nn.Module):
    def __init__(self, config, arch='bert'):
        super().__init__()
        self.arch = arch
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, arch=arch)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute", arch=arch)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config, arch=arch)
        
        if arch == 'vit':
            self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        #ltp
        self.ltp_prune_token = False
        self.ltp_prune_token_module = LTPPruneToken()

        #concrete dropout
        self.concrete_weight_regularizer = 1e-6
        self.concrete_dropout_regularizer = 1e-5
        self.concrete_calc_loss = False
        if USE_LTP_ON_CONCRETE:
            self.concrete_init_min = 0.001
            self.concrete_init_max = 0.1
        else:
            self.concrete_init_min = 0.0
            self.concrete_init_max = self.concrete_init_min
        self.concrete_prop_p_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(self.concrete_init_min, self.concrete_init_max))
        self.temperature = 0.1
        self.input_dimensionality = 0

        self.concrete_loss_factor = 1e-3

    def init_p_logits(self):
        torch.nn.init.uniform_(self.p_logit, self.concrete_init_min, self.concrete_init_max)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self.input_dimensionality = hidden_states[0].numel() # Number of elements of first item in batch

        if self.arch == 'bert':
            # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            self.self_attention_outputs = self_attention_outputs
            attention_output = self_attention_outputs[0]

            # if decoder, the last output is tuple of self-attn cache
            if self.is_decoder: raise Exception()
            else: outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            cross_attn_present_key_value = None
            if self.is_decoder and encoder_hidden_states is not None: raise Exception()

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
            if self.ltp_prune_token:
                layer_output = self.ltp_prune_token_module(layer_output, self_attention_outputs[-1], attention_mask)
            self.layer_output = layer_output
            outputs = (layer_output,) + outputs

            # if decoder, return the attn key/values as the last output
            if self.is_decoder: raise Exception()

            return outputs
        elif self.arch == 'vit':
            self_attention_outputs = self.attention(
                hidden_states=self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            # first residual connection
            hidden_states = attention_output + hidden_states

            # in ViT, layernorm is also applied after self-attention
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)

            # second residual connection is done here
            layer_output = self.output(layer_output, hidden_states)

            outputs = (layer_output,) + outputs

            return outputs
        else:
            raise Exception()

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def loss_concrete(self, input_dict):
        if self.concrete_calc_loss:
            if USE_LTP_ON_CONCRETE:
                loss = torch.mean(torch.mean(self.output.dense.concrete_mask.squeeze(-1), dim=-1) / torch.mean(input_dict['attention_mask'].squeeze(-1) * 1.0, dim = -1)) * 1e-1
            else:
                # p = torch.sigmoid(self.p_logit)

                # sum_of_square = 0
                # for param in self.parameters():
                #     sum_of_square += torch.sum(torch.pow(param, 2))
                
                # weights_regularizer = self.concrete_weight_regularizer * sum_of_square / (1 - p + EPS)
                
                # dropout_regularizer = p * torch.log(p + EPS) + (1. - p) * torch.log(1. - p + EPS)
                # dropout_regularizer *= self.concrete_dropout_regularizer * self.input_dimensionality
                
                # loss = weights_regularizer + dropout_regularizer
                loss = ((self.p_logit - self.concrete_init_min) ** 2) * self.concrete_loss_factor
                #loss = (torch.sigmoid(self.p_logit) ** 2) * 1e-6
                #raise_if_nan(loss)
                #loss = 0
            return loss
        else:
            return 0

class BertEncoder(nn.Module):
    def __init__(self, config, arch='bert'):
        super().__init__()
        self.arch = arch
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, arch=arch) for _ in range(config.num_hidden_layers)])
        for l, layer in enumerate(self.layer):
            layer.ltp_prune_token_module.init_threshold(l, config.num_hidden_layers)
        self.gradient_checkpointing = False
        self.concrete_loss_encoder_mask_avg_factor = 1e-3

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if self.arch == 'bert':
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

            next_decoder_cache = () if use_cache else None
            for i, layer_module in enumerate(self.layer):
                layer_module = layer_module # type: BertLayer
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    raise Exception()

                    if use_cache:
                        logger.warning(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                    if layer_module.ltp_prune_token:
                        attention_mask = layer_module.ltp_prune_token_module.new_attention_mask

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_decoder_cache,
                        all_hidden_states,
                        all_self_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
        elif self.arch == 'vit':
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        layer_head_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states=hidden_states, 
                        head_mask=layer_head_mask, 
                        output_attentions=output_attentions
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            raise Exception()
    
    def loss_concrete(self, input_mask):
        #assert torch.min(input_mask).item() >= 0
        layer = self.layer[0] #type: BertLayer
        target = torch.sigmoid(torch.tensor(layer.concrete_init_min, device = layer.concrete_prop_p_logit.device, dtype=torch.float32))
        occupy = 0
        count = 0
        for layer in self.layer:
            layer = layer #type: BertLayer
            if layer.output.dense.concrete_mask is not None:
                if input_mask is None:
                    occupy += torch.mean(layer.output.dense.concrete_mask)
                else:
                    occupy += \
                        torch.sum(layer.output.dense.concrete_mask.view(*input_mask.shape), dim=-1) /\
                        torch.sum(input_mask, dim=-1)
                count += 1
        occupy /= count
        occupy = occupy.mean()
        return F.mse_loss(target, occupy) * self.concrete_loss_encoder_mask_avg_factor

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    #load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

def set_print(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.get_attention().print = v

def set_masking_timing(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.get_attention().attention_masking_timing = v

def set_output_masking(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.get_attention().output_masking = v

def set_backup_last_inputs(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.get_attention().backup_last_inputs = v

class SparseBertModel(BertPreTrainedModel):

    def __init__(self, 
        config, arch='bert',
        add_pooling_layer=True
    ):
        super().__init__(config)
        self.config = config
        self.arch = arch

        if arch == 'bert':
            self.embeddings = BertEmbeddings(config)
        elif arch == 'vit':
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.embeddings = ViTEmbeddings(config)
        else:
            raise Exception('unsupported architecture', arch)
        self.encoder = BertEncoder(config, arch=arch)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        if self.arch == 'bert':
            return self.embeddings.word_embeddings
        elif self.arch == 'vit':
            return self.embeddings.patch_embeddings
        else:
            raise Exception()

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def set_print(self, v):
        for layer in self.encoder.layer:
            layer.attention.get_attention().print = v

    def set_masking_timing(self, v):
        for layer in self.encoder.layer:
            layer.attention.get_attention().attention_masking_timing = v

    def set_output_masking(self, v):
        for layer in self.encoder.layer:
            layer.attention.get_attention().output_masking = v

    def set_backup_last_inputs(self, v):
        for layer in self.encoder.layer:
            layer.attention.get_attention().backup_last_inputs = v
    
    def set_ltp_prune_token(self, v):
        for layer in self.encoder.layer:
            layer.ltp_prune_token = v
    
    def set_ltp_prune_token_soft_pruning(self, v):
        for layer in self.encoder.layer:
            layer.ltp_prune_token_module.soft_pruning = v
    
    def set_ltp_temperature(self, v):
        for layer in self.encoder.layer:
            layer = layer # type: BertLayer
            layer.ltp_prune_token_module.temperature = v

    def set_concrete_hard_threshold(self, v):
        for layer in self.encoder.layer:
            layer = layer # type: BertLayer
            layer.output.dense.concrete_hard_threshold = v
    
    def set_concrete_init_p_logit(self, v):
        for layer in self.encoder.layer:
            layer = layer # type:BertLayer
            layer.concrete_init_max = v
            layer.concrete_init_min = v
            layer.init_p_logits()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        #vit input
        pixel_values=None,
        interpolate_pos_encoding=None,
    ):
        ret = None
        timer_start('bert')
        if self.arch == 'bert':
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_attentions = True    # Always return output
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if self.config.is_decoder:
                use_cache = use_cache if use_cache is not None else self.config.use_cache
            else:
                use_cache = False

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                #print(input_ids)
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            
            if not return_dict:
                ret = (sequence_output, pooled_output) + encoder_outputs[1:]
            else:
                ret = BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )
        elif self.arch == 'vit':
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

            encoder_outputs = self.encoder(
                hidden_states=embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            sequence_output = self.layernorm(sequence_output)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            if not return_dict:
                ret = (sequence_output, pooled_output) + encoder_outputs[1:]
            else:
                ret = BaseModelOutputWithPooling(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
        else:
            raise Exception()
        
        timer_end('bert')
        return ret
        
    
    def loss_ltp_regularization(self):
        loss = 0
        for layer in self.encoder.layer:
            loss += torch.mean(layer.ltp_prune_token_module.last_mask)
        return loss

class SparseBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = SparseBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.ltp_lambda = 0.05

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if self.bert.encoder.layer[0].ltp_prune_token and (loss is not None):
            loss += self.bert.loss_ltp_regularization() * self.ltp_lambda

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ApproxBertModel(nn.Module):
    def __init__(self, origin_config, factor, wiki_train=False, arch='bert', ignore_pred=False):
        super().__init__()

        self.arch = arch
        self.factor = factor
        config = copy.deepcopy(origin_config)
        config.hidden_size = origin_config.hidden_size // self.factor
        config.intermediate_size = origin_config.intermediate_size // self.factor
        self.config = config

        if arch == 'bert':
            self.bert = SparseBertModel(config, arch=arch)
        elif arch == 'vit':
            self.bert = SparseBertModel(config, arch=arch, add_pooling_layer=False)
        else: raise Exception()
        self.bert.set_print(False)
        self.bert.set_backup_last_inputs(True)
        reset_input_mask(self.bert)
        
        if arch == 'bert':
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif arch == 'vit':
            self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        else:
            raise Exception()

        self.transfer_hidden = nn.ModuleList([
            nn.Linear(config.hidden_size, origin_config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])

        self.transfer_embedding = nn.Linear(config.hidden_size, origin_config.hidden_size)
        self.wiki_train = wiki_train
        self.ignore_pred = ignore_pred
        self.loss_att_method = 'kldiv'
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=False,
        original_output=None,
        original_emb=None,

        pixel_values=None,
        interpolate_pos_encoding=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding
        )

        if len(outputs) > 1:
            pooled_output = outputs[1]

        if self.arch == 'bert':
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif self.arch == 'vit':
            logits = self.classifier(outputs[0][:,0,:])
        else: raise Exception()

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        ret = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        if not return_loss:
            return ret
        
        approx_output = ret
        NLAYER = len(approx_output.attentions)

        # from tinybert paper
        # loss attention
        loss_att = 0
        loss_att_method = self.loss_att_method
        if loss_att_method == 'tinybert':
            if self.arch == 'bert':
                seq_len = torch.sum(attention_mask, dim=-1).to(approx_output.attentions[0].dtype)
            elif self.arch == 'vit':
                seq_len = original_output.attentions[0].shape[-1]
            else: raise Exception()
            for j in range(NLAYER):
                se = torch.square(approx_output.attentions[j] - original_output.attentions[j])
                se = se.view(se.shape[0], -1)
                se = torch.sum(se, dim=-1)
                se = se / (seq_len*seq_len)
                se = se.mean()
                loss_att += se
            loss_att /= NLAYER
            if self.arch == 'vit':
                loss_att *= 10
        elif loss_att_method == 'kldiv':
            #from miniLM paper
            N, H, T, T = approx_output.attentions[0].shape
            for j in range(NLAYER):
                y_pred = approx_output.attentions[j].view(N*H*T, T)
                y_target = original_output.attentions[j].view(N*H*T, T)
                kl_loss = y_target * ((y_target + EPS).log() - (y_pred + EPS).log())
                kl_loss = torch.sum(kl_loss.view(N, H, T, T), dim=-1) # shape: N, H, T
                if attention_mask is None:
                    kl_loss = torch.mean(kl_loss, dim=-1)
                else: # need to sum() / token_len
                    assert attention_mask.shape == (N, T) or attention_mask.shape == (1, T)
                    kl_loss = kl_loss * attention_mask.view(-1, 1, T)
                    kl_loss = torch.sum(kl_loss, dim=-1)
                    kl_loss = kl_loss / torch.sum(attention_mask, dim=-1).view(N, 1)
                    kl_loss = kl_loss
                kl_loss = kl_loss.mean() # head and batch mean
                loss_att += kl_loss
            loss_att /= NLAYER
            #loss_att *= 1/100
        else:
            raise Exception()
        
        # loss hidden
        loss_hid = 0
        for j in range(NLAYER):
            loss_hid += F.mse_loss(
                self.transfer_hidden[j](approx_output.hidden_states[j]),
                original_output.hidden_states[j]
            )
        loss_hid /= NLAYER
        if self.arch == 'vit':
            loss_hid *= (1/100)
        
        # loss emb
        if self.arch == 'bert':
            input_values = input_ids
        elif self.arch == 'vit':
            input_values = pixel_values
        else: raise Exception()
        approx_emb = self.bert.embeddings(input_values)
        loss_emb = F.mse_loss(self.transfer_embedding(approx_emb), original_emb)
        
        # loss prediction
        #print(approx_output.logits[0])
        loss_pred = F.cross_entropy(
            F.softmax(approx_output.logits, dim=-1),
            F.softmax(original_output.logits, dim=-1)
        )
        # loss_pred = F.mse_loss(
        #     F.softmax(approx_output.logits, dim=-1),
        #     F.softmax(original_output.logits, dim=-1),
        # )
        #print(approx_output.logits[0], original_output.logits[0])
        if self.arch == 'vit':
            loss_pred *= 1/10
        if self.wiki_train or self.ignore_pred:
            loss_pred *= 0

        loss = loss_att + loss_hid + loss_emb + loss_pred

        return ret, (loss, loss_att, loss_hid, loss_emb, loss_pred)

#endregion

#region Sparse Utils

def update_input_mask(sparse_bert, ks=[0.999,0.5,0.25,0.1]):
    with torch.no_grad():
        dtype = sparse_bert.encoder.layer[0].attention.get_attention().last_attention_probs.dtype
        gpu_device = sparse_bert.encoder.layer[0].attention.get_attention().last_attention_probs.device
        device = gpu_device

        batch_size, head_num, token_len, _T = sparse_bert.encoder.layer[0].attention.get_attention().last_attention_probs.shape
        assert _T == token_len
        # batch_size = sparse_bert.encoder.layer[0].attention.get_attention().last_attention_mask.shape[0]
        # token_len = sparse_bert.encoder.layer[0].attention.get_attention().last_attention_mask.shape[-1]
        
        mask = torch.zeros(batch_size, token_len, dtype=dtype, device=gpu_device)
        mask[:,0] = 1.0
        
        indices = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        impacts = torch.ones(batch_size, 1, dtype=dtype, device=device)
        benchmark_cum('mask_occupy', 1.0/token_len)

        if sparse_bert.pooler is not None:
            sparse_bert.pooler.dense.channel_indices = indices
        
        L = len(sparse_bert.encoder.layer)
        for i in range(L):
            layer = sparse_bert.encoder.layer[L-i-1]
            indices_unsqueeze = indices.unsqueeze(-1)
            layer.attention.output.dense.channel_indices = indices_unsqueeze
            layer.intermediate.dense.channel_indices = indices_unsqueeze
            layer.output.dense.channel_indices = indices_unsqueeze
            mask, indices, impacts = layer.attention.get_attention().update_input_mask_from_previous_attention(
                output_token_mask = mask,
                output_token_indices = indices,
                output_token_impact = impacts,
                k = ks[L-i-1],
            )

def reset_input_mask(sparse_bert):
    if sparse_bert.pooler is not None:
        sparse_bert.pooler.dense.channel_indices = None
        sparse_bert.pooler.dense.concrete_mask = None
    
    for layer in sparse_bert.encoder.layer:
        layer.attention.output.dense.channel_indices = None
        layer.attention.output.dense.concrete_mask = None
        
        layer.intermediate.dense.channel_indices = None
        layer.intermediate.dense.concrete_mask = None
        
        layer.output.dense.channel_indices = None
        layer.output.dense.concrete_mask = None
        
        layer.attention.get_attention().reset_input_mask()

def run_bert_with_approx(
    sparse_bert, 
    approx_bert, 
    input_dict, 
    ks=[0.5, 0.5, 0.5, 0.5],
    run_original_attention = False,
):
    timer_start('eval')
    timer_start('eval.approx_att_bert')
    with torch.no_grad():
        attention_input_dict = copy.deepcopy(input_dict)
        attention_input_dict['output_attentions'] = True
        if not run_original_attention:
            ret_approx = approx_bert(**attention_input_dict)
        else:
            reset_input_mask(sparse_bert)
            ret_approx = sparse_bert(**attention_input_dict)
        attentions = ret_approx.attentions
    timer_end('eval.approx_att_bert')

    timer_start('eval.reset_mask')
    reset_input_mask(sparse_bert)
    timer_start('eval.reset_mask.convert_mask')
    attention_mask = input_dict['attention_mask']
    if attention_mask is not None:
        attention_mask = ((1.0 - attention_mask) * (-10000)).view(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
    timer_end('eval.reset_mask.convert_mask')
    for i, layer in enumerate(sparse_bert.encoder.layer):
        layer.attention.get_attention().last_attention_probs = attentions[i]
        layer.attention.get_attention().last_approx_attention_probs = attentions[i]
        layer.attention.get_attention().last_attention_mask = attention_mask
    timer_end('eval.reset_mask')

    timer_start('eval.update_mask')
    update_input_mask(sparse_bert, ks=ks)
    timer_end('eval.update_mask')

    timer_start('eval.sparse_bert')
    ret_sparse = sparse_bert(**input_dict)
    timer_end('eval.sparse_bert')
    timer_end('eval')
    return ret_sparse
    #return ret_approx

STANDARD_NORMAL_DISTRIBUTION = torch.distributions.Normal(0, 1)

class NanException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("NaN occure")

        self.args = args

def raise_if_nan(tensor):
    return tensor
    if torch.isnan(tensor).any():
        raise NanException(tensor)
    return tensor

def run_bert_with_concrete(
    sparse_bert: "SparseBertModel",
    approx_bert: "ApproxBertModel",
    input_dict: "dict", 
):
    reset_input_mask(sparse_bert)
    #with torch.no_grad():
    attention_input_dict = copy.deepcopy(input_dict)
    attention_input_dict['output_attentions'] = True
    set_backup_last_inputs(sparse_bert, True)
    #print(attention_input_dict.keys(), 'aaaa')
    #print(attention_input_dict['pixel_values'], 'pppp')
    ret_approx = approx_bert(**attention_input_dict)
    reset_input_mask(sparse_bert)
    attentions = ret_approx.attentions
    
    attention_mask = input_dict['attention_mask']
    if attention_mask is None: #for vit compat
        scores = approx_bert.bert.encoder.layer[0].attention.get_attention().last_attention_scores
        N, H, T, _T = scores.shape
        assert T == _T
        attention_mask = torch.ones((N, T), dtype=scores.dtype, device=scores.device)
        input_dict['attention_mask'] = attention_mask
    
    attention_mask = ((1.0 - attention_mask) * (-10000)).view(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
    for i, layer in enumerate(sparse_bert.encoder.layer):
        approx_layer = approx_bert.bert.encoder.layer[i] # type: BertLayer
        layer.attention.get_attention().last_attention_mask = attention_mask
        layer.attention.get_attention().last_attention_scores = approx_layer.attention.get_attention().last_attention_scores
        layer.attention.get_attention().last_attention_probs = attentions[i]
        layer.attention.get_attention().last_approx_attention_probs = attentions[i]
        layer.attention.get_attention().last_approx_attention_score = approx_layer.attention.get_attention().last_attention_scores
        
    #update mask
    TLEN = attentions[0].shape[-1]
    N = attentions[0].shape[0]
    device = attentions[0].device
    last_score = torch.zeros((1, TLEN), dtype=torch.float32, device=device)
    last_concrete_score = torch.zeros((1, TLEN), dtype=torch.float32, device=device)
    last_mask = torch.zeros((1, TLEN), dtype=torch.float32, device=device)
    last_score[0, 0] = 1.0
    last_concrete_score[0, 0] = 1.0
    last_mask[0, 0] = 1.0
    last_mask = last_mask.expand((N, -1))
    last_mask_un = last_mask.view(-1, TLEN, 1)
    last_layer = sparse_bert.encoder.layer[-1] # type: BertLayer
    last_layer.attention.get_attention().query.concrete_mask = last_mask_un
    last_layer.attention.get_attention().query.concrete_mask_hard = last_mask_un
    last_layer.attention.output.dense.concrete_mask = last_mask_un
    last_layer.attention.output.dense.concrete_mask_hard = last_mask_un
    last_layer.intermediate.dense.concrete_mask = last_mask_un
    last_layer.intermediate.dense.concrete_mask_hard = last_mask_un
    last_layer.output.dense.concrete_mask = last_mask_un
    last_layer.output.dense.concrete_mask_hard = last_mask_un
    if BENCHMARK_CONCRETE_OCCUPY: 
        with torch.no_grad():
            benchmark_cum('concrete_occupy', last_mask_un.mean())
    last_masks = [last_mask]
    for j in range(len(sparse_bert.encoder.layer)):
        #layer indexing
        i = len(sparse_bert.encoder.layer) - j - 1
        prev_layer = None
        if i-1 >= 0:
            prev_layer = sparse_bert.encoder.layer[i-1] # type: BertLayer
        layer = sparse_bert.encoder.layer[i] # type: BertLayer

        # score calculation from STTBT
        prop_p = 0.5
        #this concrete mask is not used
        concrete_mask_mode = 'prob'
        if concrete_mask_mode == 'prob':
            att = layer.attention.get_attention().last_attention_probs
            att = torch.mean(att, dim=1)
            att = last_score.view(-1, TLEN, 1) * att * prop_p + att * (1 - prop_p)
            score = torch.mean(att, dim=1) # shape(N, TLEN)
        elif concrete_mask_mode == 'score_uniform':
            raise Exception()
            # att_mask = layer.attention.get_attention().last_attention_mask
            # onehot_att_mask = (att_mask > -1) * 1.0
            # att_score = layer.attention.get_attention().last_attention_scores
            # att_score_masked = att_score * onehot_att_mask
            # att_score_mean = torch.sum(att_score_masked, dim=-1, keepdim=True) / torch.sum(onehot_att_mask, dim=-1, keepdim=True)
            # att_score_mean_of_square = torch.sum(att_score_masked*att_score_masked, dim=-1, keepdim=True) / torch.sum(onehot_att_mask, dim=-1, keepdim=True)
            # att_score_std = torch.sqrt(att_score_mean_of_square - att_score_mean*att_score_mean)
            # std_att_score = (att_score - att_score_mean) / att_score_std
            # std_att_score = torch.mean(std_att_score, dim=1)
            # std_att_score = torch.mean(std_att_score, dim=1)
            # uni_att_score = STANDARD_NORMAL_DISTRIBUTION.cdf(std_att_score)
            # score = uni_att_score
        else:
            raise Exception()
        last_score = score
        layer.output.dense.concrete_score = last_score
        #print(score[0])
        
        if False:
            temperature = 0.01 #prev_layer.temperature
            mask = torch.sigmoid((score - layer.p_logit) / temperature) * input_dict['attention_mask']
            #prev_layer.output.dense.retain_prob = 1.0
        else:
            # accumulate dropout mask from Concrete Dropout.
            # use score as randomness source.
            concrete_score_mode = 'score_uniform'
            if concrete_score_mode == 'prob':
                concrete_score = score
                concrete_score_min = torch.min(concrete_score + (concrete_score < EPS) * 99, dim=1, keepdim=True)[0]
                concrete_score_max = torch.max(concrete_score, dim=1, keepdim=True)[0]
                concrete_score = torch.clamp_min(concrete_score - concrete_score_min + (concrete_score_max - concrete_score_min) * 0.1, 0)
                concrete_score = concrete_score / (torch.max(concrete_score, dim=1, keepdim=True)[0] + EPS)
            elif concrete_score_mode == 'score_uniform':
                #concrete_score should be unifrom distribution
                
                att_mask = layer.attention.get_attention().last_attention_mask
                N, T = input_dict['attention_mask'].shape
                onehot_att_mask = ((att_mask > -1) * 1.0) #input_dict['attention_mask'].view(N, 1, 1, T) #N,T -> N, 1, 1, T
                onehot_att_mask_sum = torch.sum(onehot_att_mask, dim=-1, keepdim=True)
                att_score = layer.attention.get_attention().last_attention_scores #N, H, T, T
                raise_if_nan(att_score)
                att_score_masked = att_score * onehot_att_mask
                raise_if_nan(att_score_masked)
                att_score_mean = torch.sum(att_score_masked, dim=-1, keepdim=True) / (onehot_att_mask_sum + EPS)
                raise_if_nan(att_score_mean)
                att_score_var = torch.sum(torch.square((att_score_masked - att_score_mean) * onehot_att_mask), dim=-1, keepdim=True) / (onehot_att_mask_sum + EPS)
                raise_if_nan(att_score_var)
                att_score_std = torch.sqrt(att_score_var)
                raise_if_nan(att_score_std)
                std_att_score = (att_score - att_score_mean) / (att_score_std + EPS)
                raise_if_nan(std_att_score)
                uni_att_score = STANDARD_NORMAL_DISTRIBUTION.cdf(std_att_score) #torch.distributions.Normal(0, 1).cdf(std_att_score)
                layer.output.dense.concrete_score_std = std_att_score
                uni_att_score = torch.mean(uni_att_score, dim=1) # head

                #uni_att_score = torch.mean(layer.attention.get_attention().last_attention_probs, dim=1)

                #std_att_score = torch.mean(std_att_score, dim=1)

                N, T, _ = uni_att_score.shape
                uni_att_score = uni_att_score * last_mask.unsqueeze(-1)
                score_prop = 0.1
                uni_att_score = torch.sum(
                    uni_att_score * last_concrete_score.unsqueeze(-1) * score_prop + uni_att_score * (1-score_prop), dim=1
                ) / (torch.sum(last_mask, dim=1, keepdim=True) + EPS)
                
                uni_att_score = uni_att_score / (torch.max(uni_att_score, dim=-1, keepdim=True)[0] + EPS)
                #uni_att_score = (0.05 + 0.95 * uni_att_score * (score > EPS)) * input_dict['attention_mask']
                empty_base = 0.01
                uni_att_score = (empty_base + (1-empty_base) * uni_att_score) * input_dict['attention_mask']
                concrete_score = uni_att_score
                last_concrete_score = concrete_score
            else:
                raise Exception()

            #print(concrete_score[0])
            p = torch.sigmoid(layer.p_logit).view(1, 1)
            layer.concrete_calc_loss = True
            temperature = layer.temperature
            layer.output.dense.concrete_score = concrete_score
            layer.output.dense.concrete_debug = [temperature, p, EPS]
            mask = torch.sigmoid((torch.log(p + EPS) - torch.log(1 - p + EPS) + torch.log(concrete_score + EPS) - torch.log(1 - concrete_score + EPS)) / (temperature))
            raise_if_nan(mask)

            #topk mask
            # topk_mask = torch.zeros_like(mask)
            # indices = torch.topk(concrete_score, k=max(1, int(concrete_score.shape[-1]*p)), dim=-1).indices
            # topk_mask = topk_mask.scatter_(-1, indices, 1.0)
            # mask = topk_mask
            #layer.output.dense.retain_prob = 1 - p
        #print(mask[0])

        concrete_hard_threshold = layer.output.dense.concrete_hard_threshold
        
        #dropout
        #mask = layer.attention.get_attention().dropout(mask) # will this help?

        #random flip
        # if layer.attention.get_attention().dropout.training:
        #     flip_prob = 0.1
        #     flip_rand = torch.rand_like(mask)
        #     flip_mask = flip_rand < flip_prob
        #     not_flip_mask = flip_rand >= flip_prob
        #     mask = mask * not_flip_mask + (1-mask) * flip_mask

        current_mask = torch.max(torch.stack([mask, last_mask], dim=0), dim=0)[0]  # this should be input mask of current layer, so set dropout mask to previous output layer.
        
        # last_masks.append(mask)
        # masks = torch.stack(last_masks, dim=-1)
        # current_mask = torch.sum(torch.softmax(masks, dim=-1) * masks, dim=-1)
        # current_mask = current_mask / (torch.max(current_mask, dim=-1, keepdim=True)[0] + EPS)
        
        current_mask_un = current_mask.view(-1, TLEN, 1)
        current_mask_hard = None
        if layer.output.dense.concrete_hard_threshold is not None:
            current_mask_hard = (current_mask_un >= concrete_hard_threshold) * 1.0
            if BENCHMARK_CONCRETE_OCCUPY:
                with torch.no_grad(): benchmark_cum('concrete_occupy', current_mask_hard.mean())
        else:
            if BENCHMARK_CONCRETE_OCCUPY:
                with torch.no_grad(): benchmark_cum('concrete_occupy', current_mask_un.mean())

        # update mask for optimization
        if prev_layer is not None:
            prev_layer.attention.get_attention().query.concrete_mask = current_mask_un
            prev_layer.attention.get_attention().query.concrete_mask_hard = current_mask_hard

            prev_layer.attention.output.dense.concrete_mask = current_mask_un
            prev_layer.attention.output.dense.concrete_mask_hard = current_mask_hard
            
            prev_layer.intermediate.dense.concrete_mask = current_mask_un
            prev_layer.intermediate.dense.concrete_mask_hard = current_mask_hard

            prev_layer.output.dense.concrete_mask = current_mask_un
            prev_layer.output.dense.concrete_mask_hard = current_mask_hard
        
        if current_mask_hard is None:
            layer.attention.get_attention().concrete_input_mask = current_mask_un.squeeze(-1)
        else:
            layer.attention.get_attention().concrete_input_mask = current_mask_hard.squeeze(-1)
        
        layer.attention.get_attention().key.concrete_mask = current_mask_un
        layer.attention.get_attention().key.concrete_mask_hard = current_mask_hard
        
        #layer.attention.get_attention().value.concrete_print = True
        layer.attention.get_attention().value.concrete_mask = current_mask_un
        layer.attention.get_attention().value.concrete_mask_hard = current_mask_hard

        # layer.attention.get_attention().query.concrete_mask = current_mask_un
        # layer.attention.get_attention().query.concrete_mask_hard = current_mask_hard

        last_mask = current_mask
    
    # forward
    ret_sparse = sparse_bert(**input_dict)
    return ret_sparse


def zero_input_mask(sparse_bert, input_dict):
    tokens = input_dict['input_ids'] #N, T
    device = tokens.device
    indices = torch.arange(tokens.shape[1], device=device, dtype=torch.int64)
    indices = indices.unsqueeze(0).repeat(*tokens.shape, 1)

    if sparse_bert.pooler is not None:
        sparse_bert.pooler.dense.channel_indices = indices.clone()
    
    for layer in sparse_bert.encoder.layer:
        layer.attention.output.dense.channel_indices = indices.clone()
        layer.intermediate.dense.channel_indices = indices.clone()
        layer.output.dense.channel_indices = indices.clone()
        layer.output.dense.concrete_mask = None
        self = layer.attention.get_attention()
        self.input_mask = None
        self.input_indices = indices.clone()
        self.input_impacts = None
        self.output_mask = None
        self.output_indices = indices.clone()
        self.query.channel_indices = indices.clone()
        self.key.channel_indices = indices.clone()
        self.value.channel_indices = indices.clone()

def run_bert_forward_sparsity(
    sparse_bert: "SparseBertModel", input_dict, ks=0.5
):
    #clear mask
    reset_input_mask(sparse_bert)
    #zero_input_mask(sparse_bert, input_dict)

    #update mask in forward path
    set_backup_last_inputs(sparse_bert, True)
    set_print(sparse_bert, False)

    #forward bert model
    with torch.no_grad():
        if sparse_bert.arch == 'bert':
            input_ids = input_dict['input_ids']
            device = input_ids.device
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            if hasattr(sparse_bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = sparse_bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            hidden_states = embedding_output = sparse_bert.embeddings(
                input_ids=input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )
            attention_mask = input_dict['attention_mask']
            extended_attention_mask = sparse_bert.get_extended_attention_mask(attention_mask, input_shape, device)
        elif sparse_bert.arch == 'vit':
            pixel_values = input_dict['pixel_values']
            hidden_states = embedding_output = sparse_bert.embeddings(
                pixel_values, 
                interpolate_pos_encoding = input_dict.get('interpolate_pos_encoding', None)
            )
            attention_mask = None
            extended_attention_mask = None
        else:
            raise Exception()

    benchmark_cum('forward_occupy', 1.0)
    #tokens = input_dict['input_ids'] #N, T
    batch_size, token_len, hidden_dim = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    indices = torch.arange(token_len, device=device, dtype=torch.int64)
    indices = indices.repeat(batch_size, 1)
    for idx, layer in enumerate(sparse_bert.encoder.layer):
        input_mask = torch.zeros(batch_size, token_len, device=device, dtype=dtype)\
            .scatter_(1, indices, 1.0)
        layer.attention.get_attention().input_mask = input_mask

        indices_unsqueeze = indices.unsqueeze(-1)
        layer.attention.get_attention().key.channel_indices = indices_unsqueeze
        layer.attention.get_attention().value.channel_indices = indices_unsqueeze
        
        with torch.no_grad(): #sparse_bert(**input_dict)
            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                None,
                None,
                None,
                None,
                True,
            )
            #hidden_states = layer_outputs[0]

        last_att = layer.attention.get_attention().last_attention_probs #(N, H, T, T)
        impact_factor = torch.mean(last_att, dim=1) #reduce head
        #impact_factor = torch.mean(impact_factor, dim=1) #reduce tokens, (N, T)
        impact_factor = torch.sum(impact_factor * input_mask.view(batch_size, token_len, 1), dim=1) / torch.sum(input_mask, dim=1, keepdim=True)
        _, indices = torch.topk(impact_factor, k=max(1, min(impact_factor.shape[1], int(ks[idx]*token_len))), dim=1)
        if idx == (len(sparse_bert.encoder.layer) - 1):
            indices = torch.zeros_like(indices)
            benchmark_cum('forward_occupy', 1.0 / token_len)
        else:
            benchmark_cum('forward_occupy', indices.shape[-1] / token_len)

        indices_unsqueeze = indices.unsqueeze(-1)
        layer.attention.output.dense.channel_indices = indices_unsqueeze
        layer.intermediate.dense.channel_indices = indices_unsqueeze
        layer.output.dense.channel_indices = indices_unsqueeze
        #layer.attention.get_attention().query.channel_indices = indices_unsqueeze 
        # this should not be able, because query layer is already used previous attention calculation...

        with torch.no_grad(): #sparse_bert(**input_dict)
            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                None,
                None,
                None,
                None,
                True,
            )
            hidden_states = layer_outputs[0]

    #run sparse
    set_backup_last_inputs(sparse_bert, False)
    ret = sparse_bert(**input_dict)
    return ret

#endregion

#region Wrappers

class ApproxSparseBertModelWrapper(nn.Module):
    def __init__(self, sparse_bert, approx_bert):
        super().__init__()
        self.sparse_bert = sparse_bert
        self.approx_bert = approx_bert
        self.run_original_attention = False

    def forward(self, input_ids, attention_mask, ks):
        output = run_bert_with_approx(
            self.sparse_bert, 
            self.approx_bert,
            {
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'output_attentions':True,
            },
            ks = ks,
            run_original_attention = self.run_original_attention
        )
        return output

class ApproxSparseBertModel(nn.Module):
    def __init__(self, bert=None, approx_bert=None, sparse_bert=None, add_pooling_layer=True, ks=0.5, arch='bert'):
        super().__init__()

        self.arch = arch

        if sparse_bert is None:
            self.sparse_bert = SparseBertModel(bert.config, add_pooling_layer=add_pooling_layer, arch=arch)
            set_print(self.sparse_bert, False)
            set_backup_last_inputs(self.sparse_bert, True)
            set_output_masking(self.sparse_bert, False)
            set_masking_timing(self.sparse_bert, 'before_softmax')
            self.sparse_bert.load_state_dict(bert.state_dict(), strict=False)
        else:
            self.sparse_bert = sparse_bert
            assert self.sparse_bert.arch == arch

        if approx_bert is None:
            self.approx_bert = ApproxBertModel(bert.config, arch=arch)
            print('approx_bert reset')
        else:
            self.approx_bert = approx_bert
            assert self.approx_bert.arch == arch
        
        if isinstance(ks, list): pass
        else: ks = [ks for _ in range(len(self.sparse_bert.encoder.layer))]
        self.ks = ks

        self.use_forward_sparse = False
        self.run_original_attention = False
        self.use_concrete_masking = False
    
    def forward(self, *args, **kwargs):
        #print(args)
        #print(kwargs['pixel_values'], 'ggifi')
        if args is not None and len(args)==1 and args[0] is not None:
            if self.arch == 'bert':
                kwargs['input_ids'] = args[0]
            elif self.arch == 'vit':
                kwargs['pixel_values'] = args[0]
            else: raise Exception()
        if 'attention_mask' not in kwargs:
            kwargs['attention_mask'] = None

        mode = kwargs.get('mode', None)
        if mode is None:
            if not self.use_forward_sparse:
                if not self.use_concrete_masking:
                    mode = 'approx'
                else:
                    mode = 'concrete'
            else:
                mode = 'forward'
                
        if mode == 'approx':
            output = run_bert_with_approx(
                sparse_bert = self.sparse_bert,
                approx_bert = self.approx_bert,
                input_dict  = kwargs,
                ks          = self.ks,
                run_original_attention = self.run_original_attention,
            )
        elif mode == 'concrete':
            #print(kwargs['pixel_values'], 'vvvv')
            output = run_bert_with_concrete(
                sparse_bert = self.sparse_bert,
                approx_bert = self.approx_bert,
                input_dict  = kwargs
            )
        elif mode == 'forward':
            output = run_bert_forward_sparsity(
                sparse_bert = self.sparse_bert,
                input_dict  = kwargs,
                ks          = self.ks,
            )
        else:
            raise Exception()

        #flops calculation
        if BENCHMARK_SPARSE_APPROX_FLOPS:
            flops_config = FlopsConfig(
                num_layer=len(self.sparse_bert.encoder.layer),
                hidden_size=self.sparse_bert.config.hidden_size,
                intermediate_size=self.sparse_bert.config.intermediate_size,
                num_heads=self.sparse_bert.config.num_attention_heads,
                seq_len=kwargs['input_ids'].shape[-1] if 'input_ids' in kwargs else 192,
                arch=self.arch,
                approx_hidden_size=self.approx_bert.config.hidden_size,
                approx_intermediate_size=self.approx_bert.config.intermediate_size,
                sparse_mode=mode,
            )
            layer_token_occupies = []
            if mode in ['approx', 'forward']:
                #from channel indices
                first_layer = self.sparse_bert.encoder.layer[0] #type: BertLayer
                layer_token_occupies.append(
                    first_layer.attention.get_attention().key.channel_indices.shape[1] / flops_config.seq_len
                )
                #print(first_layer.attention.get_attention().key.channel_indices.shape[1], flops_config.seq_len)
                for il in range(len(self.sparse_bert.encoder.layer)):
                    layer = self.sparse_bert.encoder.layer[il] #type: BertLayer
                    out_seq_len = layer.output.dense.channel_indices.shape[1]
                    layer_token_occupies.append(out_seq_len / flops_config.seq_len)
            elif mode == 'concrete':
                #from channel indices
                first_layer = self.sparse_bert.encoder.layer[0] #type: BertLayer
                N, T, _ = first_layer.attention.get_attention().key.concrete_mask_hard.shape
                assert _ == 1
                layer_token_occupies.append(
                    torch.mean(first_layer.attention.get_attention().key.concrete_mask_hard.detach().squeeze(-1), dim=-1)
                )
                #print(first_layer.attention.get_attention().key.channel_indices.shape[1], flops_config.seq_len)
                for il in range(len(self.sparse_bert.encoder.layer)):
                    layer = self.sparse_bert.encoder.layer[il] #type: BertLayer
                    layer_token_occupies.append(
                        torch.mean(layer.output.dense.concrete_mask_hard.detach().squeeze(-1), dim=-1)
                    )
            else:
                raise Exception()
            flops_config.token_occupies = layer_token_occupies
            flops = flops_sparse_approx_bert_model(flops_config)
            if isinstance(flops, torch.Tensor):
                flops = torch.mean(flops)
            #print(mode, layer_token_occupies, flops)
            #print(layer_token_occupies, flops)
            benchmark_cum('sparse_approx_flops', flops * 1e-9)
        
        return output

class ApproxSparseBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, approx_bert, arch='bert', add_pooling_layer=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.arch = arch
        
        self.bert = SparseBertModel(config, arch=arch, add_pooling_layer=add_pooling_layer)
        if arch == 'bert':
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        elif arch == 'vit':
            classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.ltp_lambda = 0.01

        self.ks = 0.5
        self.use_forward_sparse = False
        self.run_original_attention = False
        self.use_concrete_masking = False

        # Initialize weights and apply final processing
        self.post_init()

        self.approx_bert = approx_bert
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        wrapper = ApproxSparseBertModel(
            approx_bert=self.approx_bert, sparse_bert=self.bert, 
            ks=self.ks, arch=self.arch
        )
        wrapper.use_forward_sparse = self.use_forward_sparse
        wrapper.use_concrete_masking = self.use_concrete_masking
        wrapper.run_original_attention = self.run_original_attention
        outputs = wrapper(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        if self.arch == 'bert':
            pooled_output = outputs[1]
        elif self.arch == 'vit':
            pooled_output = outputs[0][:,0,:]
        else:
            raise Exception()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "custom":
                loss_fct = self.loss_fct
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        first_layer = self.bert.encoder.layer[0] # type: BertLayer
        if first_layer.ltp_prune_token and (loss is not None):
            loss += self.bert.loss_ltp_regularization() * self.ltp_lambda
        if first_layer.output.dense.concrete_mask is not None and (loss is not None):
            for layer in self.bert.encoder.layer:
                loss_reg = layer.loss_concrete({'attention_mask': attention_mask})
                loss = loss + loss_reg
            loss = loss + self.bert.encoder.loss_concrete(input_mask=attention_mask)

        ret = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

#endregion
