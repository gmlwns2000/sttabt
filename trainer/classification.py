import torch, tqdm, random
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dataset import ClassificationDataset
from models import ElectraClassifier, BigBirdClassifier

class Trainer:
    def __init__(self,
        batch_size = 16,
        device = 0,
        model = 'bigbird'
    ):
        self.seed()

        self.batch_size = batch_size
        self.device = device

        if model == 'electra':
            self.dataset = ClassificationDataset(batch_size=batch_size, tokenizer='electra')
            self.model = ElectraClassifier(self.dataset.num_classes).to(self.device)
        elif model == 'bigbird':
            self.dataset = ClassificationDataset(batch_size=batch_size, tokenizer='bigbird')
            self.model = BigBirdClassifier(self.dataset.num_classes).to(self.device)

        self.model.train()

        self.steps = 0
        self.max_steps = 360000 // self.batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        self.scaler = GradScaler()
    
    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
    
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

    def calc_acc(self, labels, output):
        indcies = torch.argmax(output, dim=1)
        acc = ((labels == indcies) * 1.0).mean()
        return acc

    def report(self, batch:ClassificationDataset):
        self.model.eval()
        
        test_batch = self.get_batch(test=True)
        with torch.no_grad(), autocast():
            tloss, toutput = self.model(test_batch, return_output=True)
            acc = self.calc_acc(batch.labels, self.output)
            tacc = self.calc_acc(test_batch.labels, toutput)

        print(f'[{self.steps}] loss:{self.loss}, acc:{acc}, test_loss:{tloss}, test_acc:{tacc}')
        self.model.train()
    
    def eval(self):
        self.model.eval()

        acc_sum = 0
        acc_count = 0
        
        for i in tqdm.tqdm(range(32000//self.batch_size)):
            test_batch = self.get_batch(test=True)
            with torch.no_grad(), autocast():
                tloss, toutput = self.model(test_batch, return_output=True)
                tacc = self.calc_acc(test_batch.labels, toutput)
                acc_sum += tacc
                acc_count += 1
        print(f'[evaluated] ({self.dataset.dataset_name}:{acc_count}*{self.batch_size}) {(acc_sum / acc_count)*100} %')
        
        self.model.train()

    def main(self):
        while self.steps < self.max_steps:
            self.steps += 1
            
            batch = self.get_batch()

            self.optimize_step(batch)

            if self.steps % 15 == 0: self.report(batch)
        
        self.eval()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()