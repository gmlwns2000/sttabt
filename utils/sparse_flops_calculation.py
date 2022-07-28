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

TOPK_FLOPS = 25
TOPK_FLOPS_LOG = 25

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

def flops_bert_embedding(c: ModelConfig):
    if c.arch == 'bert':
        #BertEmbeddings
        #nn.Embeddings does not have float operation (sparse lookup)
        flops = 2 * c.token_occupies[0] * c.seq_len * c.hidden_size
        flops += LAYER_NORM_FLOPS * c.token_occupies[0] * c.seq_len * c.hidden_size
        return flops
    elif c.arch == 'vit':
        raise Exception()

def flops_bert_self_attention(c: ModelConfig):
    flops = 0
    flops += 2*c.in_seq_len*c.hidden_size*c.hidden_size #q
    flops += 2*c.out_seq_len*c.hidden_size*c.hidden_size #k
    flops += 2*c.in_seq_len*c.hidden_size*c.hidden_size #v
    head_hidden = c.hidden_size / c.num_heads
    
    #attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    flops += c.num_heads*2*c.out_seq_len*head_hidden*c.in_seq_len

    #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    flops += c.num_heads*c.out_seq_len*c.in_seq_len
    
    #attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    flops += c.num_heads * SOFTMAX_FLOPS * c.out_seq_len * c.in_seq_len

    #context_layer = torch.matmul(attention_probs, value_layer)
    flops += 2*c.out_seq_len*c.in_seq_len*head_hidden*c.num_heads

    return flops

def flops_bert_self_output(c: ModelConfig):
    flops = 2*c.hidden_size*c.hidden_size + c.hidden_size
    if c.arch == 'bert': flops += LAYER_NORM_FLOPS * c.hidden_size + c.hidden_size
    flops *= c.out_seq_len
    return flops

def flops_bert_attention(c: ModelConfig):
    flops = flops_bert_self_attention(c)
    flops += flops_bert_self_output(c)
    return flops

def flops_bert_intermediate(c: ModelConfig):
    flops = 2*c.hidden_size*c.intermediate_size + c.intermediate_size
    flops += ACTIVATION_FLOPS*c.intermediate_size
    flops *= c.out_seq_len
    return flops

def flops_bert_output(c: ModelConfig):
    flops = 2*c.intermediate_size*c.hidden_size + c.hidden_size
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
    flops = flops_bert_embedding(c)
    flops += flops_bert_encoder(c)
    if c.arch == 'vit':
        flops += LAYER_NORM_FLOPS * c.hidden_size #pooled output
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
            flops += 2*abs(out_seq_len - in_seq_len)*c.hidden_size*c.hidden_size
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
    else:
        raise Exception()
    return flops

def flops_sparse_approx_bert_model(config:ModelConfig):
    approx_config = copy.deepcopy(config)
    approx_config.hidden_size = approx_config.approx_hidden_size
    approx_config.intermediate_size = approx_config.approx_intermediate_size
    approx_config.token_occupies = [1,] * (approx_config.num_layer+1)
    if config.sparse_mode in ['approx', 'concrete']:
        approx_flops = flops_sparse_bert_model(approx_config)
    elif config.sparse_mode == 'forward':
        approx_flops = 0
    else: raise Exception()
    update_flops = flops_sparse_update(config)
    sparse_flops = flops_sparse_bert_model(config)
    return approx_flops + sparse_flops + update_flops

def human_readable(flops):
    if flops >= 1e+15:
        return f'{flops / 1e+15:.1f} PFLOPs'
    elif flops >= 1e+12:
        return f'{flops / 1e+12:.1f} TFLOPs'
    elif flops >= 1e+9:
        return f'{flops / 1e+9:.1f} GFLOPs'
    elif flops >= 1e+6:
        return f'{flops / 1e+6:.1f} MFLOPs'
    elif flops >= 1e+3:
        return f'{flops / 1e+3:.1f} KFLOPs'

if __name__ == '__main__':
    import random
    SEQ = 512

    base_flops = flops_sparse_bert_model(ModelConfig(
        num_layer=12,
        num_heads=12,
        hidden_size=768,
        intermediate_size=768*4,
        seq_len=SEQ,
        arch='bert',
        token_occupies=None
    ))
    print('bert-base', human_readable(base_flops))

    flops = 0
    for _ in range(1000):
        factor = 8
        flops += flops_sparse_bert_model(ModelConfig(
            num_layer=12,
            num_heads=12,
            hidden_size=768/4,
            intermediate_size=768*4/4,
            seq_len=SEQ,
            arch='bert',
            token_occupies=None
        ))
    print('factor 4 approx net', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

    flops = 0
    for _ in range(1000):
        factor = 8
        flops += flops_sparse_bert_model(ModelConfig(
            num_layer=12,
            num_heads=12,
            hidden_size=768/8,
            intermediate_size=768*4/8,
            seq_len=SEQ,
            arch='bert',
            token_occupies=None
        ))
    print('factor 8 approx net', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

    flops = 0
    for _ in range(1000):
        flops += flops_sparse_update(ModelConfig(
            num_layer=12,
            num_heads=12,
            hidden_size=768,
            intermediate_size=768*4,
            approx_hidden_size=768/4,
            approx_intermediate_size=768*4/4,
            seq_len=SEQ,
            arch='bert',
            sparse_mode='approx',
            token_occupies=[random.random() * 0.4 + 0.1 for _ in range(13)],
        ))
    print('approx update', human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')

    flops = 0
    for _ in range(1000):
        flops += flops_sparse_update(ModelConfig(
            num_layer=12,
            num_heads=12,
            hidden_size=768,
            intermediate_size=768*4,
            approx_hidden_size=768/4,
            approx_intermediate_size=768*4/4,
            seq_len=SEQ,
            arch='bert',
            sparse_mode='concrete',
            token_occupies=[random.random() * 0.4 + 0.1 for _ in range(13)],
        ))
    print('concrete update', human_readable(flops / 1000), f'{flops / 1000 / base_flops * 100:.1f}%')

    flops = 0
    for _ in range(1000):
        flops += flops_sparse_update(ModelConfig(
            num_layer=12,
            num_heads=12,
            hidden_size=768,
            intermediate_size=768*4,
            approx_hidden_size=768/4,
            approx_intermediate_size=768*4/4,
            seq_len=SEQ,
            arch='bert',
            sparse_mode='forward',
            token_occupies=[random.random() * 0.4 + 0.1 for _ in range(13)],
        ))
    print('forward update', human_readable(flops / 1000), f'{flops / 1000 / base_flops * 100:.1f}%')

    for mode in ['approx', 'forward', 'concrete']:
        flops = 0
        for _ in range(1000):
            factor = 8
            flops += flops_sparse_approx_bert_model(ModelConfig(
                num_layer=12,
                num_heads=12,
                hidden_size=768,
                intermediate_size=768*4,
                seq_len=SEQ,
                arch='bert',
                sparse_mode=mode,
                approx_hidden_size=768/factor,
                approx_intermediate_size=768*factor/factor,
                token_occupies=[random.random() * 0.4 + 0.1 for _ in range(13)]
            ))
        print(mode, human_readable(flops / 1000),  f'{flops / 1000 / base_flops * 100:.1f}%')