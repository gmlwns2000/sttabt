from trainer.classification import Trainer as BaseTrainer
from transformers import BertModel
import copy, torch, math, random, time
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    def __init__(self,
        batch_size = 64,
        device = 0,
        factor = 8
    ):
        self.device = device
        self.batch_size = batch_size

        self.trainer = BaseTrainer(batch_size=batch_size, device=device)
        self.trainer.load()
        self.trainer.model.eval()
        
        self.factor = factor
        self.guide_model_method = 'new'
        
        self.bert = self.init_guide_model()
        self.bert.to(self.device)
        self.bert.train()

        self.steps = 0
        self.max_steps = 50000
        self.optimizer = optim.Adam(self.bert.parameters(), lr=5e-5)
        self.scaler = GradScaler()
    
    def init_guide_model(self):
        if self.guide_model_method == 'new':
            origin_config = self.trainer.model.bert.config
            config = copy.deepcopy(origin_config)
            config.hidden_size = origin_config.hidden_size // self.factor
            bert = BertModel(config)
            return bert
        elif self.guide_model_method == 'resize_avg':
            raise Exception()
        elif self.guide_model_method == 'resize_absmax':
            raise Exception()
        else: 
            raise Exception()

    def get_checkpoint_path(self):
        return f'att_approx_{self.factor}_{self.trainer.model_type}.pth'

    def save(self):
        torch.save({
            'bert': self.bert.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
        }, self.get_checkpoint_path())

    def load(self):
        state = torch.load(self.get_checkpoint_path(), map_location='cpu')
        self.bert.load_state_dict(state['bert'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps = state['steps']
        del state

    def report(self):
        test_batch = self.trainer.get_batch(test = True)
        self.bert.eval()
        with torch.no_grad():
            test_loss = self.calc_loss(test_batch)
        self.bert.train()
        print(f'[{self.steps}] {test_loss}({self.loss}) ({self.time_guide},{self.time_target})')

    def calc_loss(self, batch):
        target_bert = self.trainer.model.bert
        guide_bert = self.bert
        with torch.no_grad():
            t = time.time()
            target_lm_output = target_bert(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_masks, 
                output_attentions = True,
            )
            self.time_target = time.time() - t
        t = time.time()
        guide_lm_output = guide_bert(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_masks, 
            output_attentions = True,
        )
        self.time_guide = time.time() - t
        assert len(guide_lm_output.attentions) == len(target_lm_output.attentions)
        loss = 0
        for i in range(len(guide_lm_output.attentions)):
            loss += torch.mean(torch.square(guide_lm_output.attentions[i]- target_lm_output.attentions[i]))
        loss /= len(guide_lm_output.attentions)
        return loss

    def optimize_step(self, batch):
        self.optimizer.zero_grad()
        with autocast():
            self.loss = self.calc_loss(batch)
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def main(self):
        while self.steps < self.max_steps:
            batch = self.trainer.get_batch()
            self.optimize_step(batch)
            
            self.steps += 1
            if self.steps % 30 == 0: self.report()
            if self.steps % 500 == 0: self.save()
        self.save()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()