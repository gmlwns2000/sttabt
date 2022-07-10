#%%
import argparse, json
from matplotlib import pyplot as plt

import trainer.concrete_trainer as concrete

subset = 'cola'
batch_size = 4
trainer = concrete.ConcreteTrainer(
    dataset=subset,
    factor=4,
    batch_size=batch_size
)
trainer.enable_checkpointing = False

dataloader = trainer.train_dataloader

#%%
for batch in dataloader:
    batch = {k: v.to(trainer.device) for k, v in batch.items()}
    batch['output_attentions'] = True

    output = trainer.sparse_bert_inner.approx_bert(**batch)
    
    for i in range(12):
        layer = trainer.sparse_bert_inner.approx_bert.bert.encoder.layer[i] # type: concrete.sparse.BertLayer
        mask = layer.attention.self.last_attention_mask
        score = layer.attention.self.last_attention_scores
        

    break