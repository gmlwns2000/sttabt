import math
import random
import torch
from torch import optim, nn
import numpy as np
import tqdm
import transformers
from dataset.images_hf import ImagesHfDataset, ExamplesToBatchTransform, ViTInputTransform

import models.sparse_token as sparse

from utils import ddp

tasks_to_epoch = {
    'base': 100,
}

tasks_to_batch_size = {
    'base': 16
}

tasks_to_dataset = {
    'base': 'food101',
}

tasks_to_split = {
    'base': 'train',
}

model_to_hf = {
    'vit-base': 'google/vit-base-patch16-224-in21k'
}

def load_state_dict_interpolated(to_model, from_dict, ignores=['p_logit', 'ltp']):
    for name, to_param in to_model.named_parameters():
        if name in from_dict:
            from_data = from_dict[name].clone()
            to_shape = to_param.shape
            from_shape = from_data.shape
            if to_shape != from_shape:
                from_data = from_data.view(1, 1, -1)
                to_dim = 1
                for i in to_shape: to_dim *= i
                from_data = torch.nn.functional.interpolate(from_data, to_dim)
                from_data = from_data.view(*to_shape)
                to_param.data.copy_(from_data)
            else:
                to_param.data.copy_(from_data)
        else:
            ignored = False
            for ig in ignores:
                if ig in name:
                    ignored = True
                    break
            if not ignored: print(name, 'not found')

class VitApproxTrainer:
    def __init__(self,
        subset = 'base',
        model = 'vit-base',
        factor = 4,
        batch_size = -1,
        device = 0,
        world_size = 1,
        init_checkpoint=None,
    ):
        self.seed()

        self.lr = 1e-4
        self.weight_decay = 0
        self.amp_enable = True

        self.model_id = model
        self.model_id_hf = model_to_hf[model]
        self.subset = subset
        self.device = device
        self.factor = factor
        self.epochs = tasks_to_epoch[self.subset]
        if batch_size <= 0:
            batch_size = tasks_to_batch_size[self.subset]
        self.batch_size = batch_size
        self.world_size = world_size
        self.init_checkpoint = init_checkpoint

        self.init_dataloader()
        
        self.reset_train()

        if self.init_checkpoint is not None:
            self.load(self.init_checkpoint)

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
        self.extractor = transformers.ViTFeatureExtractor.from_pretrained(self.model_id_hf)
        self.dataset = ImagesHfDataset(
            ExamplesToBatchTransform(ViTInputTransform(self.extractor)),
            ExamplesToBatchTransform(ViTInputTransform(self.extractor, test=True)),
            batch_size=self.batch_size,
            name=tasks_to_dataset[self.subset],
            split=tasks_to_split[self.subset],
        )

    def get_base_model(self) -> "transformers.ViTForImageClassification":
        if self.subset == 'base':
            return transformers.ViTForImageClassification.from_pretrained(self.model_id_hf)
        else:
            raise Exception()

    def reset_train(self):
        self.seed()

        self.epoch = 0

        self.model = self.get_base_model()
        self.model = self.model.to(self.device)

        self.approx_bert = sparse.ApproxBertModel(
            self.model.config, 
            factor=self.factor, 
            arch='vit',
            ignore_pred=self.subset=='base'
        )
        self.approx_bert = self.approx_bert.to(self.device)
        load_state_dict_interpolated(self.approx_bert.bert, self.model.vit.state_dict())
        self.approx_bert = ddp.wrap_model(self.approx_bert, find_unused_paramters=True)

        self.optimizer = self.get_optimizer(self.approx_bert)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2**12, enabled=self.amp_enable)
    
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
        return f'./saves/vit-approx-{self.model_id}-{self.subset}-{self.factor}.pth'

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'approx_bert': self.approx_bert.state_dict(),

            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),

            'epoch': self.epoch,
            'epochs': self.epochs,
            'model_id': self.model_id,
            'model_id_hf': self.model_id_hf,
            'subset':self.subset,
            'factor':self.factor,
        }, self.get_checkpoint_path())
        print('VitTrainer: Checkpoint saved', self.get_checkpoint_path())
    
    def load(self, path=None):
        if path is None:
            path = self.get_checkpoint_path()
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.approx_bert.load_state_dict(state['approx_bert'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        print('VitTrainer: Checkpoint loaded', path, { 
            'epoch': state['epoch'], 
            'epochs': state['epochs'],
            'subset': state['subset'],
            'factor': state['factor']
        })
        del state

# Eval Impl

    def eval_model(self, model, approx_bert, show_message=False):
        self.model.eval()
        self.approx_bert.eval()

        pbar = tqdm.tqdm(self.dataset.get_test_iter())
        loss_sum = {'loss':0, 'loss_att':0, 'loss_hid':0, 'loss_emb':0, 'loss_pred':0, 'att_mse':0}
        count = 0
        for batch in pbar:
            batch = {k: batch[k][self.device*math.ceil(self.batch_size / self.world_size):(self.device + 1)*math.ceil(self.batch_size / self.world_size)].to(self.device, non_blocking=True) for k in batch.keys()}
            batch['output_attentions'] = True
            batch['output_hidden_states'] = True
            if 'labels' in batch: del batch['labels']

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enable):
                original_output = self.model(**batch)
                original_emb = self.model.vit.embeddings(batch['pixel_values'])
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enable):
                batch['return_loss'] = True
                batch['original_output'] = original_output
                batch['original_emb'] = original_emb

                approx_output, losses = self.approx_bert(**batch)
                loss, loss_att, loss_hid, loss_emb, loss_pred = losses
                loss_sum['loss'] += loss.item()
                loss_sum['loss_att'] += loss_att.item()
                loss_sum['loss_hid'] += loss_hid.item()
                loss_sum['loss_emb'] += loss_emb.item()
                loss_sum['loss_pred'] += loss_pred.item()
                for i in range(len(approx_output.attentions)):
                    loss_sum['att_mse'] += torch.square(approx_output.attentions[i] - original_output.attentions[i]).mean().item() / len(approx_output.attentions)
                count += 1

            pbar.set_description(f'eval [{self.epoch+1}/{self.epochs}]')
        
        for k in loss_sum.keys():
            loss_sum[k] /= count
        
        if show_message: print(loss_sum)
        return loss_sum

# Train Impl

    def train_epoch(self):
        self.model.eval()
        self.approx_bert.train()

        pbar = self.dataset.get_train_iter()
        if ddp.printable():
            pbar = tqdm.tqdm(pbar)
        
        for batch in pbar:
            batch = {k: batch[k].to(self.device, non_blocking=True) for k in batch.keys()}
            batch['output_attentions'] = True
            batch['output_hidden_states'] = True
            if 'labels' in batch: del batch['labels']

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enable):
                original_output = self.model(**batch)
                original_emb = self.model.vit.embeddings(batch['pixel_values'])
            
            with torch.cuda.amp.autocast(enabled=self.amp_enable):
                batch['return_loss'] = True
                batch['original_output'] = original_output
                batch['original_emb'] = original_emb

                approx_output, losses = self.approx_bert(**batch)
                loss, loss_att, loss_hid, loss_emb, loss_pred = losses
            
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.approx_bert.parameters(), 0.5)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if ddp.printable():
                pbar.set_description(f'[{self.epoch+1}/{self.epochs}] L:{loss.item():.5f}, Latt:{loss_att.item():.5f}, Lhid:{loss_hid.item():.5f}, Lemb:{loss_emb.item():.5f}, Lprd:{loss_pred.item():.5f}')

# Main

    def main(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            ddp.barrier()
            if ddp.printable():
                self.eval_model(self.model, self.approx_bert.module, show_message=True)
                self.save()
            ddp.barrier()

def main_ddp(rank, world_size, ddp_port, args):
    ddp.setup(rank, world_size, ddp_port)

    print('Worker:', rank, world_size, ddp_port, ddp.printable())
    trainer = VitApproxTrainer(
        device=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        subset=args.subset,
        factor=args.factor,
        init_checkpoint=args.init_checkpoint
    )
    trainer.main()

    ddp.cleanup()

def main():
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='base')
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--init-checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=-1)

    args = parser.parse_args()
    print(args)

    ddp.spawn(main_ddp, args=(args,))

if __name__ == '__main__':
    main()