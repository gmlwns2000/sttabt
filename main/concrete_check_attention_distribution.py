#%%
import argparse, json
from matplotlib import pyplot as plt

import trainer.concrete_trainer as concrete

subset = 'mnli'
batch_size = 4
trainer = concrete.ConcreteTrainer(
    dataset=subset,
    factor=4,
    batch_size=batch_size
)
trainer.enable_checkpointing = False

dataloader = trainer.train_dataloader

#%%
import torch

xs = []

for step, batch in enumerate(dataloader):
    batch = {k: v.to(trainer.device) for k, v in batch.items()}
    batch['output_attentions'] = True
    del batch['labels']

    with torch.no_grad():
        output = trainer.sparse_bert_inner.approx_bert(**batch)
        
        for i in range(12):
            layer = trainer.sparse_bert_inner.approx_bert.bert.encoder.layer[i] # type: concrete.sparse.BertLayer

            mask = layer.attention.self.last_attention_mask
            onehot_mask = (mask > -1) * 1.0
            score = layer.attention.self.last_attention_scores
            score_masked = score * onehot_mask
            score_mean = torch.sum(score_masked, dim=-1, keepdim=True) / torch.sum(onehot_mask, dim=-1, keepdim=True)
            score_mean_of_square = torch.sum(score_masked*score_masked, dim=-1, keepdim=True) / torch.sum(onehot_mask, dim=-1, keepdim=True)
            score_std = torch.sqrt(score_mean_of_square - score_mean*score_mean)
            std_score = (score - score_mean) / score_std
            std_score = torch.mean(std_score, dim=1)
            std_score = torch.mean(std_score, dim=1)

            stdnorm = torch.distributions.Normal(0, 1)
            std_score = stdnorm.cdf(std_score)
            
            prob = layer.attention.self.last_attention_probs
            for i in range(std_score.shape[0]):
                std_score_sliced = std_score[i, :int(torch.sum(onehot_mask[i,0,0]).item())].cpu().numpy()
                xs += std_score_sliced.tolist()
    if step > 300: break

plt.clf()
plt.ylim(top=len(xs))
plt.hist(xs, bins=10)
plt.show()
# %%

xs = torch.tensor(xs, dtype=torch.float32).view(1, -1)
ys = torch.distributions.Normal(0, 1).sample(xs.shape)
ys_ = torch.distributions.Normal(0.5, 1).sample(xs.shape)
print(torch.nn.functional.kl_div(ys_, ys, reduction='mean'), xs.shape)
# %%
