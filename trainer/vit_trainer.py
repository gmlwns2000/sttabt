"""
This script train
transformers.ViTForImageClassification
for specified dataset

HEEJUN LEE
"""

import random
import torch
from torch import optim, nn
import numpy as np
import tqdm
import transformers
from dataset.images_hf import ImagesHfDataset, ExamplesToBatchTransform, ViTInputTransform

tasks_to_epoch = {
    'food101_5000': 10,
    'food101': 10,
    'cifar100': 10,
    'imagenet': 10,
}

tasks_to_batch_size = {
    'food101_5000': 16,
    'food101': 16,
    'cifar100': 16,
    'imagenet': 16,
}

tasks_to_dataset = {
    'food101_5000': 'food101',
    'food101': 'food101',
    'cifar100': 'cifar100',
    'imagenet': 'imagenet-1k',
}

tasks_to_split = {
    'food101_5000': 'train[:5000]',
    'food101': 'train',
    'cifar100': 'train',
    'imagenet': 'train',
}

tasks_to_test_split = {
    'food101_5000': 'split',
    'food101': 'split',
    'cifar100': 'test',
    'imagenet': 'test',
}

tasks_to_base_model = {
    'food101_5000': 'vit-base',
    'food101': 'vit-base',
    'cifar100': 'vit-base',
    'imagenet': 'vit-base',
}

base_model_to_hf = {
    'vit-base': "google/vit-base-patch16-224-in21k",
}

class VitTrainer:
    def __init__(self,
        subset = 'cifar100',
        batch_size = -1,
        device = 0,
    ):
        self.seed()
        
        self.lr = 5e-5
        self.weight_decay = 1e-3
        self.amp_enable = True

        base_model = tasks_to_base_model[subset]
        self.base_model_id = base_model
        self.base_model_id_hf = base_model_to_hf[base_model]
        self.subset = subset
        self.device = device
        self.epochs = tasks_to_epoch[self.subset]
        self.device = device
        if batch_size <= 0:
            batch_size = tasks_to_batch_size[self.subset]
        self.batch_size = batch_size

        self.init_dataloader()
        
        self.reset_train()

# Initialize

    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_dataloader(self):
        self.extractor = transformers.ViTFeatureExtractor.from_pretrained(self.base_model_id_hf)
        self.dataset = ImagesHfDataset(
            ExamplesToBatchTransform(ViTInputTransform(self.extractor)),
            ExamplesToBatchTransform(ViTInputTransform(self.extractor, test=True)),
            batch_size=self.batch_size,
            name=tasks_to_dataset[self.subset],
            split=tasks_to_split[self.subset],
            test_split=tasks_to_test_split[self.subset],
        )

    def reset_train(self):
        self.seed()

        self.epoch = 0

        self.model = transformers.ViTForImageClassification.from_pretrained(
            self.base_model_id_hf,
            num_labels=self.dataset.num_labels,
            id2label=self.dataset.id2label,
            label2id=self.dataset.label2id
        )
        self.model = self.model.to(self.device)

        self.optimizer = self.get_optimizer(self.model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enable)
    
    def get_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        kwargs = {
            'lr':self.lr,
            'weight_decay':self.weight_decay,
        }
        
        return optim.AdamW(params, **kwargs)

# IO

    def get_checkpoint_path(self):
        return f'./saves/vit-base-{self.subset}.pth'

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'subset': self.subset,
            'base_model_id': self.base_model_id,
            'base_model_id_hf': self.base_model_id_hf,
            'epoch': self.epoch,
            'epochs': self.epochs,
        }, self.get_checkpoint_path())
        print('VitTrainer: Checkpoint saved', self.get_checkpoint_path())
    
    def load(self):
        state = torch.load(self.get_checkpoint_path(), map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        del state
        print('VitTrainer: Checkpoint loaded', self.get_checkpoint_path())

# Eval Impl

    def eval_model(self, model, show_message=False):
        model.eval()
        from datasets import load_metric

        metric = load_metric("accuracy")

        pbar = tqdm.tqdm(self.dataset.get_test_iter(), desc='eval')
        for batch in pbar:
            batch = {k: batch[k].to(self.device, non_blocking=True) for k in batch.keys()}
            labels = batch['labels']
            del batch['labels']

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enable):
                output = model(**batch)

            metric.add_batch(predictions=torch.argmax(output[0], dim=-1), references=labels)
        score = metric.compute()
        if show_message: print(score)
        return score

# Train Impl

    def train_epoch(self):
        self.model.train()

        pbar = tqdm.tqdm(self.dataset.get_train_iter())
        for batch in pbar:
            batch = {k: batch[k].to(self.device, non_blocking=True) for k in batch.keys()}
            
            with torch.cuda.amp.autocast(enabled=self.amp_enable):
                output = self.model(**batch)
                loss = output.loss
            
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            pbar.set_description(f'[{self.epoch+1}/{self.epochs}] {loss.item():.5f}')

# Main

    def main(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            self.eval_model(self.model, show_message=True)
            self.save()

def main():
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='cifar100')
    parser.add_argument('--batch-size', type=int, default=-1)

    args = parser.parse_args()

    trainer = VitTrainer(
        batch_size=args.batch_size,
        subset=args.subset
    )
    trainer.main()

if __name__ == '__main__':
    main()