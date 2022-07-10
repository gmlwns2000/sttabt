#%%
import trainer.concrete_trainer as concrete

trainer = concrete.ConcreteTrainer(
    'cola',
    factor=4,
)

#%%
trainer.sparse_bert.module.bert.set_concrete_init_p_logit(3332.0)
trainer.sparse_bert.module.bert.set_concrete_hard_threshold(0.5)
concrete.sparse.benchmark_reset()
trainer.eval_sparse_model()
print('occ', concrete.sparse.benchmark_get_average('concrete_occupy'))
# %%
trainer.reset_train()
trainer.sparse_bert.module.bert.set_concrete_init_p_logit(4.0)
trainer.sparse_bert.module.bert.set_concrete_hard_threshold(None)
trainer.sparse_bert.module.use_concrete_masking = True
concrete.sparse.benchmark_reset()
print(trainer.eval_sparse_model())
print('occ', concrete.sparse.benchmark_get_average('concrete_occupy'))
# %%
l = trainer.sparse_bert.module.bert.encoder.layer[-2] #type: concrete.sparse.BertLayer
l.output.dense.concrete_mask_hard[3].view(-1)
# %%
