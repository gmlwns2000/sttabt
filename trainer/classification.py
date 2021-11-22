import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from dataset import ClassificationDataset
from models import ElectraClassifier

class Trainer:
    def __init__(self,
        batch_size = 32,
        device = 0,
    ):
        self.batch_size = batch_size
        self.device = device

        self.dataset = ClassificationDataset(batch_size=batch_size)
        
        self.model = ElectraClassifier(self.dataset.num_classes).to(self.device)
        self.model.train()

        self.steps = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        self.scaler = GradScaler()
    
    def get_batch(self, test=False):
        batch = self.dataset.batch(test=test)
        return batch.to(self.device)

    def optimize_step(self, batch:ClassificationDataset):
        self.optimizer.zero_grad()
        with autocast():
            self.loss, self.output = self.model(batch, return_output=True)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def report(self, batch:ClassificationDataset):
        def calc_acc(labels, output):
            indcies = torch.argmax(output, dim=1)
            acc = ((labels == indcies) * 1.0).mean()
            return acc

        self.model.eval()
        
        test_batch = self.get_batch(test=True)
        with torch.no_grad(), autocast():
            tloss, toutput = self.model(test_batch, return_output=True)
            acc = calc_acc(batch.labels, self.output)
            tacc = calc_acc(test_batch.labels, toutput)

        print(f'[{self.steps}] loss:{self.loss}, acc:{acc}, test_loss:{tloss}, test_acc:{tacc}')
        self.model.train()

    def main(self):
        while True:
            self.steps += 1
            
            batch = self.get_batch()

            self.optimize_step(batch)

            if self.steps % 15 == 0: self.report(batch)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()