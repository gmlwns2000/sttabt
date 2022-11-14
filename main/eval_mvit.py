from models import mvit_timm
import tqdm, torch
from trainer.vit_approx_trainer import VitApproxTrainer

model = mvit_timm.mvitv2_tiny_sttabt(pretrained=True)
model = model.to(0)
model.eval()

trainer = VitApproxTrainer(model='deit-small')
acc = 0.0
c = 0
for i, batch in enumerate(tqdm.tqdm(trainer.timm_data_test)):
    batch = {'pixel_values': batch[0].to(trainer.device), 'labels': batch[1].to(trainer.device)}
    inp = batch['pixel_values']
    label = batch['labels']
    
    def accuracy(logits, labels):
        return ((torch.argmax(logits, dim=-1) == labels)*1.0).mean().item()
    
    with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
        acc += accuracy(model(inp)['logits'], label)
    c += 1

print('Acc:', acc/c)