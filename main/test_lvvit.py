import transformers, torch, copy
config = transformers.ViTConfig(
    hidden_size = 384, num_hidden_layers = 16, num_attention_heads = 6, 
    intermediate_size = 384*3, hidden_act = 'gelu', hidden_dropout_prob = 0.0, 
    attention_probs_dropout_prob = 0.0,initializer_range = 0.02,
    layer_norm_eps = 1e-12,is_encoder_decoder = False,image_size = 224,
    patch_size = 16,num_channels = 3,qkv_bias = False,encoder_stride = 16,
    num_labels=1000
)
vit = transformers.ViTForImageClassification(config).eval()

from models import lvvit
lvmodel = lvvit.lvvit_s().eval()
state = torch.load('./saves/lvvit/lvvit_s-26M-224-83.3.pth.tar', map_location='cpu')
lvmodel.load_state_dict(state)

# print('-'*80)
# print(vit.state_dict().keys())
# print('-'*80)
# print(lvmodel.state_dict().keys())
# print('-'*80)

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

new_state = transform_dict_lv_to_vit(state)
vit.vit.embeddings.patch_embeddings = copy.deepcopy(lvmodel.patch_embed)
vit.vit.embeddings.patch_embeddings.sttabt_patched = True
vit.load_state_dict(new_state)

img = torch.randn((1,3,224,224))
outlv = lvmodel(img)
outvit = vit(img)['logits']
print(outlv[0,:10], outvit[0,:10])

# t = torch.randn((1, 16*16, 384))
# for i in range(16):
#     print((vit.vit.encoder(t)[0] - lvmodel.blocks[i](t)).mean())

def init_concrete(model):
    import models.sparse_token as sparse
    import trainer.vit_approx_trainer as vit_approx

    approx_bert = sparse.ApproxBertModel(
        model.config, factor=4, wiki_train=False, arch='vit', ignore_pred=True
    )

    concrete_model = sparse.ApproxSparseBertForSequenceClassification(
        model.config,
        approx_bert,
        arch = 'vit',
        add_pooling_layer=False,
    )
    assert hasattr(concrete_model.bert.encoder, 'concrete_loss_encoder_mask_avg_factor')
    concrete_model.bert.encoder.concrete_loss_encoder_mask_avg_factor = 1.0
    for layer in concrete_model.bert.encoder.layer:
        assert hasattr(layer, 'concrete_loss_factor')
        layer.concrete_loss_factor = 1e-1
    concrete_model.bert.embeddings.patch_embeddings = copy.deepcopy(model.vit.embeddings.patch_embeddings)
    concrete_model.bert.embeddings.patch_embeddings.sttabt_patched = True
    
    try:
        concrete_model.bert.load_state_dict(
            vit_approx.get_vit(model).state_dict(),
            strict=True,
        )
    except Exception as ex:
        print('load vit', ex)

    try:
        concrete_model.classifier.load_state_dict(
            model.classifier.state_dict(),
            strict=True,
        )
    except Exception as ex:
        print('load classifier', ex)
    
    concrete_model.to(0)#.train()
    concrete_model.use_concrete_masking = True
    #concrete_model = ddp.wrap_model(self.concrete_model, find_unused_paramters=True)
    concrete_model#.train()

    # self.set_concrete_init_p_logit(self.init_p_logit)
    # self.set_concrete_hard_threshold(None)
    concrete_model.bert.set_concrete_hard_threshold(None)
    concrete_model.bert.set_concrete_init_p_logit(10000.0) #full mask

    return concrete_model.eval() #always eval

import tqdm
from trainer.vit_approx_trainer import VitApproxTrainer
trainer = VitApproxTrainer(model='deit-small')
lvmodel = lvmodel.to(trainer.device).eval()
vit = vit.to(trainer.device).eval()
cmodel = init_concrete(vit)

cacc = 0.0
vacc = 0.0
lvacc = 0.0

c = 0
for i, batch in enumerate(tqdm.tqdm(trainer.timm_data_test)):
    if i > 100: break

    batch = {'pixel_values': batch[0].to(trainer.device), 'labels': batch[1].to(trainer.device)}
    inp = batch['pixel_values']
    label = batch['labels']
    
    def accuracy(logits, labels):
        return ((torch.argmax(logits, dim=-1) == labels)*1.0).mean().item()
    
    with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
        cacc += accuracy(cmodel(inp)['logits'], label)
        vacc += accuracy(vit(inp)['logits'], label)
        lvacc += accuracy(lvmodel(inp), label)
    c += 1
print('loaded vit, wo aux', vacc/c, 'lvvit', lvacc/c, 'concerete loaded, wo aux', cacc/c)