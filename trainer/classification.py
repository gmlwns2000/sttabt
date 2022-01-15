import torch, tqdm, random
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from dataset import ClassificationDataset
from models import ElectraClassifier, BigBirdClassifier, BertClassification

class Trainer:
    def __init__(self,
        batch_size = 64,
        device = 0,
        model = 'bert-mini'
    ):
        self.seed()

        self.batch_size = batch_size
        self.device = device
        
        self.model_type = model
        if model == 'electra':
            self.dataset = ClassificationDataset(batch_size=batch_size, tokenizer='electra')
            self.model = ElectraClassifier(self.dataset.num_classes).to(self.device)
        elif model == 'bigbird':
            self.dataset = ClassificationDataset(batch_size=batch_size, tokenizer='bigbird')
            self.model = BigBirdClassifier(self.dataset.num_classes).to(self.device)
        elif model == 'bert-mini':
            self.dataset = ClassificationDataset(batch_size=batch_size, tokenizer='bert')
            self.model = BertClassification(self.dataset.num_classes).to(self.device)
        elif model == 'bert-base':
            self.dataset = ClassificationDataset(batch_size=batch_size, tokenizer='bert')
            self.model = BertClassification(self.dataset.num_classes,
                bert_model_name = 'google/bert_uncased_L-12_H-768_A-12'
            ).to(self.device)
        else:
            raise Exception('unknown model')

        print(f'Trainer.__init__: Model initialized. model = {model}')
        self.model.train()

        self.steps = 0
        self.max_steps = 360000 // self.batch_size  # 3 epoch on AG NEWS
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        self.scaler = GradScaler()
    
    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
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

        print(f'[{self.steps}/{self.max_steps}] loss:{self.loss}, acc:{acc}, test_loss:{tloss}, test_acc:{tacc}')
        self.model.train()
    
    def eval(self):
        self.model.eval()
        self.seed()

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
    
    def get_checkpoint_path(self):
        return f'saves/cls_{self.model_type}.pth'

    def save(self):
        print('Trainer.save: Saving...', self.get_checkpoint_path())
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
        }, self.get_checkpoint_path())
    
    def load(self):
        print('Trainer.load: Loading...', self.get_checkpoint_path())
        state = torch.load(self.get_checkpoint_path(), map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps = state['steps']
        del state

    def main(self):
        while self.steps < self.max_steps:
            self.steps += 1
            
            batch = self.get_batch()

            self.optimize_step(batch)

            if self.steps % 15 == 0: self.report(batch)
            if self.steps % 300 == 0: self.save()
        
        self.save()
        self.eval()

if __name__ == '__main__':
    trainer = Trainer(
        model = 'bert-base',
        batch_size = 16, 
    )
    trainer.main()