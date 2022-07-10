from trainer.classification import Trainer
from trainer.attention_approx import Trainer as ApproxTrainer
import models.sparse_token as sparse
import importlib, torch, math

batch_size= 8 
device = 1
factor = 16

trainer = Trainer(device=device, batch_size=batch_size, model='bert-mini')
trainer.model.eval()
trainer.load()
bert = trainer.model.bert
fc = trainer.model.classifier
batch = trainer.get_batch().to(device)
test_batch = trainer.get_batch(test=True).to(device)

approx_trainer = ApproxTrainer(model=trainer.model_type, device=trainer.device, batch_size=batch_size, factor=factor)
approx_trainer.bert.eval()
approx_trainer.load()
approx_bert = approx_trainer.bert

sparse_bert = sparse.SparseBertModel(bert.config)
sparse_bert.to(trainer.device)
sparse_bert.eval()
sparse_bert.load_state_dict(bert.state_dict())
sparse.set_print(sparse_bert, False)

ks = [int(math.ceil(batch.input_ids.shape[1] * 0.5))] * len(sparse_bert.encoder.layer)
jitmodel = sparse.ApproxSparseBertModel(sparse_bert, approx_bert)
jit_result = torch.jit.trace(jitmodel, (batch.input_ids, batch.attention_masks, ks))
jit_result(batch.input_ids, batch.attention_masks, ks)
print('done')