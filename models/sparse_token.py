import math, os, torch, numba, time, datetime
from shutil import ExecError
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
import torch.nn.functional as F

logger = logging.get_logger(__name__)

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

#region Model Definition

#old version

#static K version
def update_input_mask_from_previous_attention(
    attention_mask, 
    previous_attention, 
    output_token_indices, 
    output_token_impact, 
    head_reduce_method = 'avg',
    token_reduce_method = 'avg',
    apply_token_impact = True,
    k=0.5,
    k_estimate = True,
    k_estimate_threshold = 0.1,
):
    attention_mask_shape = attention_mask.shape
    NBATCH = attention_mask_shape[0]
    TLEN = attention_mask_shape[-1]
    dtype = previous_attention.dtype
    device = previous_attention.device
    if k < 1.0:
        kxx = int(math.ceil(k*TLEN))
    else:
        kxx = k
    
    if head_reduce_method == 'avg':
        att = torch.sum(previous_attention, dim=1)  #reduce head
    elif head_reduce_method == 'max':
        att = torch.max(previous_attention, dim=1)[0]
    else: raise Exception()

    att = torch.gather(att, 1, output_token_indices.unsqueeze(-1).expand(NBATCH, output_token_indices.shape[1], TLEN))
    if apply_token_impact: 
        att = att * output_token_impact.unsqueeze(-1)
    
    if token_reduce_method == 'avg':
        att = torch.sum(att, dim=1)                 #reduce token
    elif token_reduce_method == 'max':
        att = torch.max(att, dim=1)[0]
    else: raise Exception()
    #att(N, TLEN)
    if k_estimate:
        est_k = max(1, math.ceil(torch.max(torch.sum((att > k_estimate_threshold) * 1.0, dim=1)).item()))
        #print(kxx, est_k, TLEN)
        kxx = est_k
    input_impacts, input_indices = torch.topk(att, kxx, dim=1)
    #input_impacts(N, K), input_indices(N, K)
    
    input_mask = torch.zeros(NBATCH, TLEN, device=device, dtype=dtype)\
        .scatter_(1, input_indices, 1.0)
    #input_mask (N, TLEN)
    
    return input_mask, input_indices, input_impacts

from transformers.models.bert.modeling_bert import BertEmbeddings

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

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.channel_indices is None or self.force_dense:
            timer_start('sparselinear.force.linear')
            x = F.linear(input, self.weight, self.bias)
            timer_end('sparselinear.force.linear')
            return x
        else:
            timer_start('sparselinear')
            input_shape = input.shape
            if len(input_shape) != 3:
                print(input_shape)
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

            timer_start('sparselinear.scatter_')
            x = torch.zeros(N, C, self.out_features, 
                dtype=x.dtype, device=x.device).\
                    scatter_(dim=1, index=channel_indices_unsqueeze.expand(-1,-1,self.out_features), src=x)
            timer_end('sparselinear.scatter_')
            timer_end('sparselinear')
            return x
            #return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = SparseChannelLinear(config.hidden_size, self.all_head_size)
        self.key = SparseChannelLinear(config.hidden_size, self.all_head_size)
        self.value = SparseChannelLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise Exception('removed')

        self.is_decoder = config.is_decoder
        if self.is_decoder: raise Exception()

        self.print = True
        self.reset_input_mask()
        self.attention_masking_timing = 'before_softmax'
        self.output_masking = False
        self.backup_last_inputs = True

        #self.query.force_dense = self.key.force_dense = self.value.force_dense = False
    
    def reset_input_mask(self):
        self.input_mask = None
        self.input_indices = None
        self.input_impacts = None
        self.output_mask = None
        self.output_indices = None
    
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
        self.query.channel_indices = output_token_indices.unsqueeze(-1)
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
            attention_scores = attention_scores + (1.0 - self.input_mask.view(self.input_mask.shape[0], 1, 1, self.input_mask.shape[-1])) * -10000

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        timer_start('bert.attention.probs.dropout')
        attention_probs = self.dropout(attention_probs)
        timer_end('bert.attention.probs.dropout')

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        if self.backup_last_inputs:
            if self.print: print('SelfAttention.forward: last_attention_probs backuped')
            self.last_attention_probs = attention_probs.detach().clone()
            self.last_attention_mask = attention_mask.detach().clone()

        if self.input_mask is not None and self.attention_masking_timing == 'after_softmax':
            if self.print: print(f'apply input mask, after softmax. input_mask:{self.input_mask.shape}, attention_probs:{attention_probs.shape}')
            attention_probs = attention_probs * self.input_mask.view(self.input_mask.shape[0], 1, 1, self.input_mask.shape[-1])

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
    def __init__(self, config):
        super().__init__()
        self.dense = SparseChannelLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
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
        self_outputs = self.self(
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
    def __init__(self, config):
        super().__init__()
        self.dense = SparseChannelLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        timer_start('bert.output')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        timer_end('bert.output')
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

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
        self.layer_output = layer_output
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder: raise Exception()

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

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
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

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


class SparseBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

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
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

#endregion

#region Sparse Utils

def set_print(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.self.print = v

def set_masking_timing(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.self.attention_masking_timing = v

def set_output_masking(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.self.output_masking = v

def set_backup_last_inputs(sparse_bert, v):
    for layer in sparse_bert.encoder.layer:
        layer.attention.self.backup_last_inputs = v

def update_input_mask(sparse_bert, ks=[0.999,0.5,0.25,0.1]):
    with torch.no_grad():
        dtype = sparse_bert.encoder.layer[0].attention.self.last_attention_probs.dtype
        gpu_device = sparse_bert.encoder.layer[0].attention.self.last_attention_probs.device
        device = gpu_device

        batch_size = sparse_bert.encoder.layer[0].attention.self.last_attention_mask.shape[0]
        token_len = sparse_bert.encoder.layer[0].attention.self.last_attention_mask.shape[-1]
        
        mask = torch.zeros(batch_size, token_len, dtype=dtype, device=gpu_device)
        mask[:,0] = 1.0
        
        indices = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        impacts = torch.ones(batch_size, 1, dtype=dtype, device=device)

        if sparse_bert.pooler is not None:
            sparse_bert.pooler.dense.channel_indices = indices
        
        L = len(sparse_bert.encoder.layer)
        for i in range(L):
            layer = sparse_bert.encoder.layer[L-i-1]
            indices_unsqueeze = indices.unsqueeze(-1)
            layer.attention.output.dense.channel_indices = indices_unsqueeze
            layer.intermediate.dense.channel_indices = indices_unsqueeze
            layer.output.dense.channel_indices = indices_unsqueeze
            mask, indices, impacts = layer.attention.self.update_input_mask_from_previous_attention(
                output_token_mask = mask,
                output_token_indices = indices,
                output_token_impact = impacts,
                k = ks[L-i-1],
            )

def reset_input_mask(sparse_bert):
    for layer in sparse_bert.encoder.layer:
        layer.attention.self.reset_input_mask()

def run_bert_with_approx(
    sparse_bert, 
    approx_bert, 
    input_dict, 
    ks=[0.5, 0.5, 0.5, 0.5]
):
    timer_start('approx_att')
    with torch.no_grad():
        ret_approx = approx_bert(**input_dict)
        attentions = ret_approx.attentions
    timer_end('approx_att')

    timer_start('approx_mask_reset')
    reset_input_mask(sparse_bert)
    timer_start('approx_mask_reset_inverse')
    attention_mask = input_dict['attention_mask']
    attention_mask = ((1.0 - attention_mask) * (-10000)).view(attention_mask.shape[0], 1, 1, attention_mask.shape[-1])
    timer_end('approx_mask_reset_inverse')
    for i, layer in enumerate(sparse_bert.encoder.layer):
        layer.attention.self.last_attention_probs = attentions[i]
        layer.attention.self.last_attention_mask = attention_mask
    timer_end('approx_mask_reset')

    timer_start('approx_mask_update')
    update_input_mask(sparse_bert, ks=ks)
    timer_end('approx_mask_update')

    timer_start('approx_sparse')
    ret_sparse = sparse_bert(**input_dict)
    timer_end('approx_sparse')
    return ret_sparse
    return ret_approx

class ApproxSparseBertModel(nn.Module):
    def __init__(self, sparse_bert, approx_bert):
        super().__init__()
        self.sparse_bert = sparse_bert
        self.approx_bert = approx_bert

    def forward(self, input_ids, attention_mask, ks):
        output = run_bert_with_approx(
            self.sparse_bert, 
            self.approx_bert,
            {
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'output_attentions':True,
            },
            ks = ks
        )
        return output

#endregion