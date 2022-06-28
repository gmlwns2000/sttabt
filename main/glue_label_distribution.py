import trainer.ltp_trainer as ltp
import torch

tokenizer = ltp.transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

dataloader = ltp.get_dataloader('mrpc', tokenizer, 64, split='validation')

x = 0
c = 0
for batch in dataloader:
    label = batch['labels']
    if label.shape[0] == 64:
        x += label
        c += 1

print(torch.mean(x/c, dim=0))