import random
import torch
from torch import optim, nn
import numpy as np
import tqdm
import transformers
from dataset.images_hf import ImagesHfDataset, ExamplesToBatchTransform, ViTInputTransform

tasks_to_epoch = {
    'food101_5000': 10,
    'imagenet': 10,
}

tasks_to_batch_size = {
    'food101_5000': 16,
    'imagenet': 16,
}

tasks_to_dataset = {
    'food101_5000': 'food101',
    'imagenet': 'imagenet-1k',
}

tasks_to_split = {
    'food101_5000': 'train[:5000]',
    'imagenet': 'train',
}

class VitTrainer:
    def __init__(self,
        subset = 'food101_5000',
        batch_size = -1,
        device = 0,
    ):
        self.lr = 1e-5
        self.weight_decay = 1e-3
        self.amp_enable = True

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
        self.extractor = transformers.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dataset = ImagesHfDataset(
            ExamplesToBatchTransform(ViTInputTransform(self.extractor)),
            ExamplesToBatchTransform(ViTInputTransform(self.extractor, test=True)),
            batch_size=self.batch_size,
            name=tasks_to_dataset[self.subset],
            split=tasks_to_split[self.subset],
        )

    def reset_train(self):
        self.seed()

        self.model = transformers.ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
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

def main():
    trainer = VitTrainer()
    trainer.main()

if __name__ == '__main__':
    main()