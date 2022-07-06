from trainer.glue_base import GlueAttentionApproxTrainer as Trainer
import models.sparse_token as sparse
from torch import optim
import torch

trainer = Trainer('cola', 4, batch_size=128, enable_plot=False)
model = sparse.ApproxSparseBertForSequenceClassification(trainer.model.config, approx_bert=trainer.approx_bert)
model.load_state_dict(trainer.model.state_dict(), strict=False)
# for name, p in model.named_parameters():
#     if name.find('p_logit') >= 0:
#         p.requires_grad = True
#     else:
#         p.requires_grad = False
model.to(trainer.device).train()
model.use_concrete_masking = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    for batch in trainer.train_dataloader:
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        batch['output_attentions'] = True

        model.bert.set_concrete_hard_threshold(0.5)
        output = model(**batch)
        loss = output['loss']
        # print(loss)
        # print('>>', end='')
        # input()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    for layer in model.bert.encoder.layer:
        layer = layer # type: sparse.BertLayer
        print(torch.mean(torch.mean(layer.attention.self.last_attention_scores, dim=1), dim=1)[0])
        score = torch.mean(torch.mean(layer.attention.self.last_approx_attention_probs, dim=1), dim=1)
        print(score[0], layer.attention.self.last_approx_attention_probs.shape)
        if layer.output.dense.concrete_mask is not None:
            print(layer.output.dense.concrete_score[0])
            print(layer.output.dense.concrete_mask[0].squeeze(-1))
            print(layer.p_logit.item(), torch.sigmoid(layer.p_logit).item())
    print('>>', end='')
    input()