#%%
from trainer.glue_base import GlueAttentionApproxTrainer as Trainer
import models.sparse_token as sparse
from torch import optim
import torch

trainer = Trainer('cola', 4, batch_size=64, enable_plot=False)
trainer.load()
model = sparse.ApproxSparseBertForSequenceClassification(trainer.model.config, approx_bert=trainer.approx_bert.module)
model.load_state_dict(trainer.model.state_dict(), strict=False)
model.to(trainer.device).train()
model.use_concrete_masking = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()
use_autocast = False

#%%
def debug():
    masks = []
    for layer in model.bert.encoder.layer:
        layer = layer # type: sparse.BertLayer
        # score = torch.mean(torch.mean(layer.attention.self.last_approx_attention_score, dim=1), dim=1)
        # print('ascore', score[0], layer.attention.self.last_approx_attention_score.shape)
        if layer.output.dense.concrete_mask is not None:
            #print('cdebug', layer.output.dense.concrete_debug)
            print('cscore', layer.output.dense.concrete_score[0])
            if layer.output.dense.concrete_mask_hard is not None:
                masks.append(layer.output.dense.concrete_mask_hard[0].squeeze(-1))
            else:
                masks.append(layer.output.dense.concrete_mask[0].squeeze(-1))
            #print('cmask', layer.output.dense.concrete_mask[0].squeeze(-1))
            #print('a', layer.p_logit.item(), torch.sigmoid(layer.p_logit).item())
    masks = torch.round(torch.stack(masks), decimals=2)
    print(masks)

sparse.benchmark_reset()
model.bert.set_concrete_init_p_logit(-1)
model.bert.set_concrete_hard_threshold(0.5)
print(trainer.eval_base_model(model))
model.bert.set_concrete_hard_threshold(None)
print('concrete occupy', sparse.benchmark_get_average('concrete_occupy'))
debug()
print('>>', end='')
input()

#%%
from matplotlib import pyplot as plt
plt.clf()

plt.show()

#%%
for epoch in range(50):
    for batch in trainer.train_dataloader:
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        batch['output_attentions'] = True
        
        with torch.cuda.amp.autocast(enabled = use_autocast):
            model.train()
            model.bert.set_concrete_hard_threshold(None)
            output = model(**batch)
            loss = output['loss']
            print(loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled = use_autocast):
            model.eval()
            model.bert.set_concrete_hard_threshold(0.5)
            _output = model(**batch)
            _ = _output['loss']
    
    debug()

    print('>>', end='')
    input()