"""
cite: https://github.com/google-research/electra/blob/master/flops_computation.py
returns inference flops
batch size always 1
"""

import math, copy, random

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5

SIGMOID_FLOPS = 5

LOG_FLOPS = 5

CDF_FLOPS = 10

TOPK_FLOPS = 35
TOPK_FLOPS_LOG = 35

MM_FLOPS = 2

class ModelConfig:
    def __init__(self,
        num_layer,
        hidden_size,
        intermediate_size,
        num_heads,
        seq_len,
        arch,
        sparse_mode=None,
        approx_hidden_size=None,
        approx_intermediate_size=None,
        token_occupies=None,
        patch_size=16,
        dyvit_pruning_loc=[3,6,9],
        patch_embeding_mode = 'vit',
        image_size = 224,
    ):
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.seq_len = seq_len
        self.arch = arch
        self.sparse_mode = sparse_mode
        self.approx_hidden_size = approx_hidden_size
        self.approx_intermediate_size = approx_intermediate_size
        if token_occupies is None:
            token_occupies = [1,]*(num_layer+1)
        self.token_occupies = token_occupies
        self.in_seq_len = None
        self.out_seq_len = None
        self.patch_size = patch_size
        self.dyvit_pruning_loc = dyvit_pruning_loc
        self.patch_embeding_mode = patch_embeding_mode
        self.image_size = image_size

from typing import List
from numpy import prod

#https://github.com/facebookresearch/fvcore/blob/34cceb47e0298a7bebee8491328ab2c983c7a0a2/fvcore/nn/jit_handles.py#L129
def flops_conv_op(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
):
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape) * MM_FLOPS
    return flop

def flops_bert_embedding(c: ModelConfig):
    if c.arch == 'bert':
        #BertEmbeddings
        #nn.Embeddings does not have float operation (sparse lookup)

        #encoding addition
        flops = 2 * c.token_occupies[0] * c.seq_len * c.hidden_size
        flops += LAYER_NORM_FLOPS * c.token_occupies[0] * c.seq_len * c.hidden_size
        return flops
    elif c.arch == 'vit':
        if c.patch_embeding_mode == 'vit':
            #ViTEmbeddings
            #vit embedding use conv fc.
            #assume there is no patch embedding resize
            flops = MM_FLOPS * c.token_occupies[0] * c.seq_len * (c.patch_size ** 2) * c.hidden_size
            flops += c.token_occupies[0] * c.seq_len * c.hidden_size
            flops += LAYER_NORM_FLOPS * c.token_occupies[0] * c.seq_len * c.hidden_size
            return flops
        elif c.patch_embeding_mode == 'lvvit':
            #LVViT patch embedding is different.
            #lvvit_layers.PatchEmbed4_2
            isize = c.image_size
            flops = 0
            
            # conv must perform regardless token drops...
            # self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
            flops += flops_conv_op((1, 3, isize, isize), (3,64,7,7), (1,64,isize,isize))
            # self.bn1 = nn.BatchNorm2d(64)
            flops += prod((1,64,isize,isize)) * 2
            # self.relu = nn.ReLU(inplace=True)
            flops += prod((1,64,isize,isize))
            
            # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
            flops += flops_conv_op((1, 64, isize, isize), (64,64,3,3), (1,64,isize,isize))
            # self.bn2 = nn.BatchNorm2d(64)
            flops += prod((1,64,isize,isize)) * 2
            flops += prod((1,64,isize,isize))

            # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  
            flops += flops_conv_op((1, 64, isize, isize), (64,64,3,3), (1,64,isize,isize))
            # self.bn3 = nn.BatchNorm2d(64)
            flops += prod((1,64,isize,isize)) * 2
            flops += prod((1,64,isize,isize))

            # self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
            flops = MM_FLOPS * c.token_occupies[0] * c.seq_len * (c.patch_size ** 2) * c.hidden_size
            return flops
        else: raise Exception()
    else: raise Exception()

def flops_bert_self_attention(c: ModelConfig):
    flops = 0
    flops += MM_FLOPS*c.in_seq_len*c.hidden_size*c.hidden_size #q
    flops += MM_FLOPS*c.out_seq_len*c.hidden_size*c.hidden_size #k
    flops += MM_FLOPS*c.in_seq_len*c.hidden_size*c.hidden_size #v
    head_hidden = c.hidden_size / c.num_heads
    
    #attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    flops += c.num_heads*MM_FLOPS*c.out_seq_len*head_hidden*c.in_seq_len

    #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    flops += c.num_heads*c.out_seq_len*c.in_seq_len
    
    #attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    flops += c.num_heads * SOFTMAX_FLOPS * c.out_seq_len * c.in_seq_len

    #context_layer = torch.matmul(attention_probs, value_layer)
    flops += MM_FLOPS*c.out_seq_len*c.in_seq_len*head_hidden*c.num_heads

    return flops

def flops_bert_self_output(c: ModelConfig):
    flops = MM_FLOPS*c.hidden_size*c.hidden_size + c.hidden_size
    if c.arch == 'bert':
        flops += LAYER_NORM_FLOPS * c.hidden_size + c.hidden_size
    flops *= c.out_seq_len
    return flops

def flops_bert_attention(c: ModelConfig):
    flops = flops_bert_self_attention(c)
    flops += flops_bert_self_output(c)
    return flops

def flops_bert_intermediate(c: ModelConfig):
    flops = MM_FLOPS*c.hidden_size*c.intermediate_size + c.intermediate_size
    flops += ACTIVATION_FLOPS*c.intermediate_size
    flops *= c.out_seq_len
    return flops

def flops_bert_output(c: ModelConfig):
    flops = MM_FLOPS*c.intermediate_size*c.hidden_size + c.hidden_size
    flops += c.hidden_size
    if c.arch == 'bert': flops += c.hidden_size * LAYER_NORM_FLOPS
    flops *= c.out_seq_len
    return flops

def flops_bert_layer(c: ModelConfig):
    flops = flops_bert_attention(c)
    flops += flops_bert_intermediate(c)
    flops += flops_bert_output(c)
    if c.arch == 'vit':
        #in layer norm
        flops += LAYER_NORM_FLOPS * c.in_seq_len * c.hidden_size
        #out layer norm
        flops += LAYER_NORM_FLOPS * c.out_seq_len * c.hidden_size
        #res conn
        flops += c.out_seq_len * c.hidden_size
    # TODO! LTPPruneToken handling
    return flops

def flops_bert_encoder(c: ModelConfig):
    flops = 0
    for ilayer in range(c.num_layer):
        c.in_seq_len = c.seq_len * c.token_occupies[ilayer]
        c.out_seq_len = c.seq_len * c.token_occupies[ilayer+1]
        flops += flops_bert_layer(c)
    return flops

def flops_sparse_bert_model(c: ModelConfig):
    assert len(c.token_occupies) == (c.num_layer+1)

    global LAYER_NORM_FLOPS, \
        ACTIVATION_FLOPS, \
        SOFTMAX_FLOPS, \
        SIGMOID_FLOPS, \
        LOG_FLOPS, \
        CDF_FLOPS, \
        TOPK_FLOPS, \
        TOPK_FLOPS_LOG, MM_FLOPS

    if c.arch == 'vit':
        #https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py#L31
        #we need to fuse the operations, ignore operations
        prev_values = (
            MM_FLOPS, LOG_FLOPS, SIGMOID_FLOPS, CDF_FLOPS, 
            SOFTMAX_FLOPS, ACTIVATION_FLOPS, LAYER_NORM_FLOPS
        )
        MM_FLOPS = 1    #fvcore use fused mm
        LOG_FLOPS = 0
        SIGMOID_FLOPS = 0
        CDF_FLOPS = 0
        SOFTMAX_FLOPS = 0
        ACTIVATION_FLOPS = 0
        LAYER_NORM_FLOPS = 5
        # we do not ignore topk, since for topk is not negligible

    flops = flops_bert_embedding(c)
    flops += flops_bert_encoder(c)
    if c.arch == 'vit':
        flops += LAYER_NORM_FLOPS * c.hidden_size #pooled output
    
    if c.arch == 'vit':
        (
            MM_FLOPS, LOG_FLOPS, SIGMOID_FLOPS, CDF_FLOPS, 
            SOFTMAX_FLOPS, ACTIVATION_FLOPS, LAYER_NORM_FLOPS
        ) = prev_values

    return flops

def flops_sparse_update(c:ModelConfig):
    flops = 0
    if c.sparse_mode == 'approx':
        for ilayer in range(c.num_layer):
            out_seq_len = c.token_occupies[ilayer+1]*c.seq_len
            flops += (c.num_heads + 1)*out_seq_len*c.seq_len
            flops += (out_seq_len + 1)*c.seq_len
            flops += TOPK_FLOPS * c.seq_len + TOPK_FLOPS_LOG * math.log(c.seq_len)
            flops += c.seq_len * 4 #mask accumulation could be implemented in linear time
            in_seq_len = c.token_occupies[ilayer]*c.seq_len
            flops += in_seq_len
    elif c.sparse_mode == 'forward':
        for ilayer in range(c.num_layer):
            in_seq_len = c.token_occupies[ilayer]*c.seq_len
            out_seq_len = c.token_occupies[ilayer+1]*c.seq_len
            #unmaks query
            flops += MM_FLOPS*abs(out_seq_len - in_seq_len)*c.hidden_size*c.hidden_size
            flops += (c.num_heads + 1)*c.seq_len*in_seq_len
            flops += (c.seq_len + 1)*in_seq_len
            flops += TOPK_FLOPS * in_seq_len + TOPK_FLOPS_LOG * math.log(in_seq_len)
    elif c.sparse_mode == 'concrete':
        for ilayer in range(c.num_layer):
            out_seq_len = c.token_occupies[ilayer+1]*c.seq_len
            #attention shape (out, in)
            flops += c.num_heads*8*out_seq_len*c.seq_len #standardization
            flops += c.num_heads*CDF_FLOPS*out_seq_len*c.seq_len
            flops += (c.num_heads+1)*out_seq_len*c.seq_len
            # uni_att_score = torch.sum(
            #     uni_att_score * last_concrete_score.unsqueeze(-1) * score_prop + uni_att_score * (1-score_prop), dim=1
            # ) / (torch.sum(last_mask, dim=1, keepdim=True) + EPS)
            # now, reduced
            flops += 5*out_seq_len*c.seq_len
            #uni_att_score = uni_att_score / (torch.max(uni_att_score, dim=-1, keepdim=True)[0] + EPS)
            flops += 2*c.seq_len
            #uni_att_score = (empty_base + (1-empty_base) * uni_att_score) * input_dict['attention_mask']
            flops += 2*c.seq_len
            #mask = torch.sigmoid((torch.log(p + EPS) - torch.log(1 - p + EPS) + torch.log(concrete_score + EPS) - torch.log(1 - concrete_score + EPS)) / (temperature))
            flops += (SIGMOID_FLOPS + 2 * LOG_FLOPS + 3)*c.seq_len
            flops += c.seq_len * (c.num_layer - ilayer)
            flops += c.seq_len
    elif c.sparse_mode == 'dyvit':
        for ilayer in range(c.num_layer):
            if ilayer in c.dyvit_pruning_loc:
                in_seq_len = c.token_occupies[ilayer]*c.seq_len
                #calculate in_conv
                flops += MM_FLOPS*in_seq_len*c.hidden_size*c.hidden_size
                flops += LAYER_NORM_FLOPS*in_seq_len*c.hidden_size
                #calculate global_x
                flops += in_seq_len*c.hidden_size/2*2
                #calculate out_conv
                flops += MM_FLOPS*in_seq_len*c.hidden_size*(c.hidden_size/2)
                flops += LAYER_NORM_FLOPS*in_seq_len*c.hidden_size/2
                flops += MM_FLOPS*in_seq_len*(c.hidden_size/2)*(c.hidden_size/4)
                flops += LAYER_NORM_FLOPS*in_seq_len*c.hidden_size/4
                flops += MM_FLOPS*in_seq_len*(c.hidden_size/4)*2
                flops += SOFTMAX_FLOPS*in_seq_len*2
    else:
        raise Exception()
    return flops

def flops_sparse_approx_bert_model(config:ModelConfig):
    global LAYER_NORM_FLOPS, \
        ACTIVATION_FLOPS, \
        SOFTMAX_FLOPS, \
        SIGMOID_FLOPS, \
        LOG_FLOPS, \
        CDF_FLOPS, \
        TOPK_FLOPS, \
        TOPK_FLOPS_LOG, MM_FLOPS

    if config.arch == 'vit':
        #https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py#L31
        #we need to fuse the operations, ignore operations
        prev_values = (
            MM_FLOPS, LOG_FLOPS, SIGMOID_FLOPS, CDF_FLOPS, 
            SOFTMAX_FLOPS, ACTIVATION_FLOPS, LAYER_NORM_FLOPS
        )
        MM_FLOPS = 1
        LOG_FLOPS = 0
        SIGMOID_FLOPS = 0
        CDF_FLOPS = 0
        SOFTMAX_FLOPS = 0
        ACTIVATION_FLOPS = 0
        LAYER_NORM_FLOPS = 5

    
    approx_config = copy.deepcopy(config)
    approx_config.hidden_size = approx_config.approx_hidden_size
    approx_config.intermediate_size = approx_config.approx_intermediate_size
    approx_config.token_occupies = [1,] * (approx_config.num_layer+1)
    if config.sparse_mode in ['approx', 'concrete']:
        approx_flops = flops_sparse_bert_model(approx_config)
    elif config.sparse_mode in ['forward', 'dyvit']:
        approx_flops = 0
    else: raise Exception()
    update_flops = flops_sparse_update(config)
    sparse_flops = flops_sparse_bert_model(config)

    if config.arch == 'vit':
        (
            MM_FLOPS, LOG_FLOPS, SIGMOID_FLOPS, CDF_FLOPS, 
            SOFTMAX_FLOPS, ACTIVATION_FLOPS, LAYER_NORM_FLOPS
        ) = prev_values
    
    return approx_flops + sparse_flops + update_flops

def human_readable(flops):
    if flops < 0:
        return f'-{human_readable(-flops)}'
    if flops >= 1e+15:
        return f'{flops / 1e+15:.1f} PFLOPs'
    elif flops >= 1e+12:
        return f'{flops / 1e+12:.1f} TFLOPs'
    elif flops >= 1e+9:
        return f'{flops / 1e+9:.2f} GFLOPs'
    elif flops >= 1e+6:
        return f'{flops / 1e+6:.1f} MFLOPs'
    elif flops >= 1e+3:
        return f'{flops / 1e+3:.1f} KFLOPs'
    else:
        return f'{flops:.1f} FLOPs'

if __name__ == '__main__':
    import random
    SEQ = 128

    def exam(base_config):
        base_flops = flops_sparse_bert_model(base_config)
        print('bert', human_readable(base_flops))

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.hidden_size /= 4
            config.intermediate_size /= 4
            flops += flops_sparse_bert_model(config)
        print('factor 4 approx net', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.hidden_size /= 4
            config.intermediate_size = config.hidden_size
            flops += flops_sparse_bert_model(config)
        print('factor 4 tiny approx net', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.hidden_size /= 8
            config.intermediate_size /= 8
            flops += flops_sparse_bert_model(config)
        print('factor 8 approx net', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.hidden_size /= 8
            config.intermediate_size /= config.hidden_size
            flops += flops_sparse_bert_model(config)
        print('factor 8 tiny approx net', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.approx_hidden_size = config.hidden_size / 4
            config.approx_intermediate_size = config.intermediate_size / 4
            config.sparse_mode = 'approx'
            config.token_occupies = [random.random() * 0.4 + 0.1 for _ in range(config.num_layer+1)]
            flops += flops_sparse_update(config)
        print('approx update', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.approx_hidden_size = config.hidden_size / 4
            config.approx_intermediate_size = config.intermediate_size / 4
            config.sparse_mode = 'concrete'
            config.token_occupies = [random.random() * 0.4 + 0.1 for _ in range(config.num_layer+1)]
            flops += flops_sparse_update(config)
        print('concrete update', human_readable(flops / 1000), f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.approx_hidden_size = config.hidden_size / 4
            config.approx_intermediate_size = config.intermediate_size / 4
            config.sparse_mode = 'forward'
            config.token_occupies = [random.random() * 0.4 + 0.1 for _ in range(config.num_layer+1)]
            flops += flops_sparse_update(config)
        print('forward update', human_readable(flops / 1000), f'{flops / 1000 / base_flops * 100:.1f}%')

        flops = 0
        for _ in range(1000):
            config = copy.deepcopy(base_config)
            config.approx_hidden_size = config.hidden_size / 4
            config.approx_intermediate_size = config.intermediate_size / 4
            config.sparse_mode = 'dyvit'
            config.token_occupies = [random.random() * 0.4 + 0.1 for _ in range(config.num_layer+1)]
            flops += flops_sparse_update(config)
        print('dyvit update', human_readable(flops / 1000), f'{flops / 1000 / base_flops * 100:.1f}%')

        for mode in ['approx', 'forward', 'concrete', 'dyvit']:
            flops = 0
            for _ in range(1000):
                factor = 8
                config = copy.deepcopy(base_config)
                config.approx_hidden_size = config.hidden_size / factor
                config.approx_intermediate_size = config.intermediate_size / factor
                config.sparse_mode = mode
                config.token_occupies = [random.random() * 0.4 + 0.1 for _ in range(config.num_layer+1)]
                flops += flops_sparse_approx_bert_model(config)
            print(mode, factor, human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')
    
    base_config = ModelConfig(
        num_layer=12,
        num_heads=12,
        hidden_size=768,
        intermediate_size=768*4,
        seq_len=SEQ,
        arch='bert',
        token_occupies=None
    )
    print('-'*80)
    print('bert-base')
    print('-'*80)
    exam(base_config)

    large_config = ModelConfig(
        num_layer=24,
        num_heads=16,
        hidden_size=1024,
        intermediate_size=1024*4,
        seq_len=SEQ,
        arch='bert',
        token_occupies=None
    )
    print('-'*80)
    print('bert-large')
    print('-'*80)
    exam(large_config)

    base_config = ModelConfig(
        num_layer=12,
        num_heads=12,
        hidden_size=768,
        intermediate_size=768*4,
        seq_len=14*14+1,
        arch='vit',
        token_occupies=None
    )
    print('-'*80)
    print('deit-base@fp32')
    print('-'*80)
    exam(base_config)

    base_config = ModelConfig(
        num_layer=12,
        num_heads=3,
        hidden_size=192,
        intermediate_size=192*4,
        seq_len=14*14+1,
        arch='vit',
        token_occupies=None
    )
    print('-'*80)
    print('deit-tiny@fp32')
    print('-'*80)
    exam(base_config)

    base_config = ModelConfig(
        num_layer=12,
        num_heads=6,
        hidden_size=384,
        intermediate_size=384*4,
        seq_len=14*14+1,
        arch='vit',
        token_occupies=None
    )
    print('-'*80)
    print('deit-small@fp32')
    print('-'*80)
    exam(base_config)