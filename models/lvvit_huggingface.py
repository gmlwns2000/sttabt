from models import lvvit
import transformers, torch, copy

def transform_dict_lv_to_vit(state, hidden_size=384, skip_lam = 2.0):
    new_state = {}
    replaces = {
        'blocks.': 'vit.encoder.layer.',
        'cls_token': 'vit.embeddings.cls_token',
        'pos_embed': 'vit.embeddings.position_embeddings',
        'patch_embed.': 'vit.embeddings.patch_embeddings.',
        '.mlp.fc1.': '.intermediate.dense.',
        '.mlp.fc2.': '.output.dense.',
        '.attn.proj.': '.attention.output.dense.',
        'head.': 'classifier.',
        'norm.': 'vit.layernorm.',
        '.norm1.': '.layernorm_before.',
        '.norm2.': '.layernorm_after.',
    }
    remove = ['aux_head.bias', 'aux_head.weight']
    for key in state.keys():
        old_key = key
        if not any([k == key for k in remove]):
            for k in replaces.keys():
                if k in key:
                    key = key.replace(k, replaces[k])
            assert not key in new_state
            new_state[key] = state[old_key]
    qkv_state = {}
    for key in new_state.keys():
        if '.attn.qkv.' in key:
            h = hidden_size
            param = new_state[key]
            assert param.shape == (h*3, h)
            qkv_state[key.replace('.attn.qkv.', '.attention.attention.query.')] = param[:h]
            qkv_state[key.replace('.attn.qkv.', '.attention.attention.key.')] = param[h:h*2]
            qkv_state[key.replace('.attn.qkv.', '.attention.attention.value.')] = param[h*2:h*3]
        else:
            qkv_state[key] = new_state[key]
    
    for key in qkv_state.keys():
        if ('output.dense.weight' in key or 'output.dense.bias' in key) and\
            (not 'attention' in key):
            # print(key)
            qkv_state[key] *= 1/skip_lam
        elif ('attention.output.dense.weight' in key or 'attention.output.dense.bias' in key):
            # print(key)
            qkv_state[key] *= 1/skip_lam
    return qkv_state

def load_model(modelid):
    print('LvvitHuggingface: model id', modelid)
    if modelid == 'lvvit-small':
        config = transformers.ViTConfig(
            hidden_size = 384, num_hidden_layers = 16, num_attention_heads = 6, 
            intermediate_size = 384*3, hidden_act = 'gelu', hidden_dropout_prob = 0.0, 
            attention_probs_dropout_prob = 0.0,initializer_range = 0.02,
            layer_norm_eps = 1e-12,is_encoder_decoder = False,image_size = 224,
            patch_size = 16,num_channels = 3,qkv_bias = False,encoder_stride = 16,
            num_labels=1000
        )
        vit = transformers.ViTForImageClassification(config).eval()
        
        lvmodel = lvvit.lvvit_s().eval()
        state = torch.load(
            './saves/lvvit/lvvit_s-26M-224-83.3.pth.tar', 
            map_location='cpu'
        )
        lvmodel.load_state_dict(state)

        new_state = transform_dict_lv_to_vit(state)
        vit.vit.embeddings.patch_embeddings = copy.deepcopy(lvmodel.patch_embed)
        vit.vit.embeddings.patch_embeddings.sttabt_patched = True
        vit.load_state_dict(new_state)

        return vit
    else:
        raise Exception()