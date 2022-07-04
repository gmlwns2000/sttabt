from trainer.glue_base import GlueAttentionApproxTrainer as Trainer
import models.sparse_token as sparse
from torch import optim

trainer = Trainer('cola', 8, batch_size=64, enable_plot=False)
model = sparse.ApproxSparseBertForSequenceClassification(trainer.model.config, approx_bert=trainer.approx_bert)
model.load_state_dict(trainer.model.state_dict(), strict=False)
model.to(trainer.device).train()
model.use_concrete_masking = True

optimizer = optim.Adam(model.parameters(), lr=2e-5)

for batch in trainer.train_dataloader:
    batch = {k: v.to(trainer.device) for k, v in batch.items()}
    batch['output_attentions'] = True

    output = model(**batch)
    loss = output['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())