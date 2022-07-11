#%%
import trainer.concrete_trainer as concrete

trainer = concrete.ConcreteTrainer(
    'mrpc',
    factor=4,
)

#%%
trainer.sparse_bert.module.bert.set_concrete_init_p_logit(-0.0)
trainer.sparse_bert.module.bert.set_concrete_hard_threshold(0.5)
concrete.sparse.benchmark_reset()
trainer.eval_sparse_model()
print('occ', concrete.sparse.benchmark_get_average('concrete_occupy'))
# %%
trainer.reset_train()
trainer.sparse_bert.module.bert.set_concrete_init_p_logit(-0.5)
trainer.sparse_bert.module.bert.set_concrete_hard_threshold(None)
trainer.sparse_bert.module.use_concrete_masking = True
concrete.sparse.benchmark_reset()
print(trainer.eval_sparse_model())
print('occ', concrete.sparse.benchmark_get_average('concrete_occupy'))
# %%
l = trainer.sparse_bert.module.bert.encoder.layer[-2] #type: concrete.sparse.BertLayer
print(
    l.output.dense.concrete_score[0].view(-1), 
    l.output.dense.concrete_mask[0].view(-1), 
    (l.output.dense.concrete_mask[0].view(-1) > 0.5) * 1,
    sep='\n'
)
# %%
import torch, tqdm
from matplotlib import pyplot as plt
model = trainer.sparse_bert_inner
model.eval()
scores = []
for step, batch in enumerate(tqdm.tqdm(trainer.test_dataloader)):
    batch = {k: v.to(trainer.device) for k, v in batch.items()}
    batch['output_attentions'] = True
    del batch['labels']
    
    with torch.no_grad():
        model(**batch)

    mask = batch['attention_mask']
    for il, layer in enumerate(model.bert.encoder.layer):
        if il == (len(model.bert.encoder.layer) - 1): continue
        layer = layer #type: concrete.sparse.BertLayer
        for ib in range(mask.shape[0]):
            slice = layer.output.dense.concrete_score[ib, :int(torch.sum(mask[ib]).item())].view(-1)
            scores.append(slice)

scores = torch.concat(scores)
scores = scores.cpu().numpy()

plt.clf()
plt.ylim(top=len(scores)/5)
plt.hist(scores, bins=50)
plt.show()
# %%
