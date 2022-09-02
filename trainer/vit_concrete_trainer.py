import gc, json
import math
import random
import torch
torch.set_num_threads(1)
from torch import optim, nn
import numpy as np
import tqdm
import transformers
from dataset.images_hf import ImagesHfDataset, ExamplesToBatchTransform, ViTInputTransform

import models.sparse_token as sparse

from trainer import vit_approx_trainer as vit_approx
from utils import ddp

tasks_to_epoch = {
    'base': 30,
    'cifar100': 20,
    'imagenet': 30,
}

tasks_to_batch_size = {
    'vit-base': 16,
    'deit-base': 16,
    'deit-small': 32,
}

def log(*args):
    print("VitConcreteTrainer:", *args)

class VitConcreteTrainer:
    def __init__(self,
        subset = 'base',
        model = 'deit-small',
        factor = 4,
        batch_size = -1,
        device = 0,
        world_size = 1,
        init_checkpoint=None,
        enable_valid = False,
        epochs = None,
        init_p_logit = 0.0,
        json_postfix = None,
    ):
        self.device = device
        if batch_size < 0:
            self.batch_size = vit_approx.tasks_to_batch_size[model]
        else:
            self.batch_size = batch_size
        self.factor = factor
        self.world_size = world_size
        self.model_id = model
        self.json_postfix = json_postfix

        self.enable_valid = enable_valid
        self.lr = 1e-5
        self.weight_decay = 5e-2
        self.init_p_logit = init_p_logit
        self.epochs = tasks_to_epoch[subset] if epochs is None else epochs
        self.batch_size = tasks_to_batch_size[model] if batch_size <= 0 else batch_size

        self.approx_trainer = vit_approx.VitApproxTrainer(
            subset = subset,
            model = model,
            factor = factor,
            batch_size = self.batch_size,
            device = device,
            world_size = world_size,
            init_checkpoint = init_checkpoint,
            enable_valid = self.enable_valid,
        )
        #self.approx_trainer.init_dataloader("split" if self.enable_valid else "val")
        assert self.approx_trainer.dataloader_lib == 'timm'
        try:
            self.approx_trainer.load()
        except Exception as ex:
            log(ex)
            log("Failed to load attention approximation!")
        
        self.init_concrete()
        
        self.init_optim()

        self.tqdm_position = 0
        self.tqdm_prefix = ''
        self.enable_checkpointing = True
    
    def init_concrete(self):
        self.concrete_model = sparse.ApproxSparseBertForSequenceClassification(
            self.approx_trainer.model.config,
            self.approx_trainer.approx_bert.module,
            arch = 'vit',
            add_pooling_layer=False,
        )
        
        try:
            self.concrete_model.bert.load_state_dict(
                vit_approx.get_vit(self.approx_trainer.model).state_dict(),
                strict=True,
            )
        except Exception as ex:
            log('load vit', ex)
        
        try:
            self.concrete_model.classifier.load_state_dict(
                self.approx_trainer.model.classifier.state_dict(),
                strict=True,
            )
        except Exception as ex:
            log('load classifier', ex)
        
        self.concrete_model.to(self.device).train()
        self.concrete_model.use_concrete_masking = True
        self.concrete_model = ddp.wrap_model(self.concrete_model, find_unused_paramters=True)

        self.set_concrete_init_p_logit(self.init_p_logit)
        self.set_concrete_hard_threshold(None)
    
    def set_concrete_init_p_logit(self, v):
        self.concrete_model.module.bert.set_concrete_init_p_logit(v)
    
    def set_concrete_hard_threshold(self, v):
        self.concrete_model.module.bert.set_concrete_hard_threshold(v)
    
    def init_optim(self):
        self.steps = 0
        self.epoch = 0
        self.last_metric_score = None
        self.last_loss = None

        self.optimizer = self.get_optimizer(self.concrete_model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    def get_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        high_lr = ['p_logit']
        params = [
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in high_lr))], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (any(nd in n for nd in high_lr))], 'lr':self.lr * 1.0, 'weight_decay': 0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        #print(params[1])

        kwargs = {
            'lr':self.lr,
            'weight_decay':self.weight_decay,
        }
        
        return optim.AdamW(params, **kwargs)

    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# eval function

    def eval_concrete(self, show_message=True, tqdm_prefix=''):
        from datasets import load_metric

        # check average loss
        sparse.benchmark_concrete_occupy(True)
        self.concrete_model.eval()
        ddp_model = self.concrete_model
        self.concrete_model = ddp_model.module

        loss_sum = 0
        loss_count = 0
        metric = load_metric("accuracy")
        sparse.benchmark_reset()
        for i, batch in enumerate(tqdm.tqdm(self.approx_trainer.timm_data_test, desc=f'{self.tqdm_prefix}{tqdm_prefix}eval', position=self.tqdm_position)):
            if i > 50: break
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled = False):
                #batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                batch = {'pixel_values': batch[0], 'labels': batch[1]} #timm compatibility
                batch['output_attentions'] = True
                output = self.concrete_model(**batch)
                #output = self.approx_trainer.model(**batch)
                loss = output.loss
            metric.add_batch(predictions=torch.argmax(output[1], dim=-1), references=batch['labels'])
            
            loss_sum += loss.item()
            loss_count += 1
        score = metric.compute()
        score['valid_loss'] = loss_sum / loss_count
        score['occupy'] = sparse.benchmark_get_average('concrete_occupy')
        if show_message: log('eval score:', score)

        self.concrete_model = ddp_model
        self.concrete_model.train()

        return score

# train function

    def train_eval(self):
        self.set_concrete_hard_threshold(None)
        soft_result = self.eval_concrete(tqdm_prefix = 'soft prune ')
        
        self.set_concrete_hard_threshold(0.5)
        hard_result = self.eval_concrete(tqdm_prefix = 'hard prune ')

        self.set_concrete_hard_threshold(None)

        json_path = f'./saves_plot/vit-concrete-eval{("-"+self.json_postfix) if self.json_postfix is not None else ""}.json'
        with open(json_path, 'w') as f:
            log('Json dumped', json_path)
            json.dump({
                'epoch': self.epoch,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'world_size': self.world_size,
                'dataset': self.approx_trainer.subset,
                'init_p_logit': self.init_p_logit,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'enable_valid': self.enable_valid,

                'soft_result': soft_result,
                'hard_result': hard_result,

                'previous_results': self.train_eval_previous_results,
            }, f, indent=2)
            self.train_eval_previous_results.append({
                'epoch': self.epoch,
                'train_mask_method': self.train_mask_method,
                'soft_result': soft_result,
                'hard_result': hard_result,
            })

    def train_epoch(self):
        sparse.benchmark_concrete_occupy(False)
        pbar = self.approx_trainer.timm_data_train
        if ddp.printable():
            pbar = tqdm.tqdm(pbar, position = self.tqdm_position)
        
        for istep, batch in enumerate(pbar):
            if istep > 200: break

            batch = {'pixel_values': batch[0], 'labels': batch[1]} #timm compatibility
            batch['output_attentions'] = True
            batch['output_hidden_states'] = True

            with torch.cuda.amp.autocast(enabled=False):
                output = self.concrete_model(**batch)
                loss = output.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.concrete_model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.last_loss = loss.item()

            if ddp.printable():
                pbar.set_description(f"{self.tqdm_prefix}[{self.epoch+1}/{self.epochs}, {self.train_mask_method}] L:{self.last_loss:.5f}")
            
            self.steps += 1

    def main(self):
        self.train_mask_method = 'soft'
        self.train_eval_previous_results = []

        #before train
        self.epoch = -1
        if ddp.printable() and self.enable_checkpointing:
            self.train_eval()
        if self.world_size > 1:
            ddp.dist.barrier()
        
        #after train
        for epoch in range(self.epochs):
            self.epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
        
            if epoch >= min(self.epochs - 1, (self.epochs - 1) * 0.8):
                self.set_concrete_hard_threshold(0.5)
                self.train_mask_method = 'hard'
            else:
                self.set_concrete_hard_threshold(None)
                self.train_mask_method = 'soft'

            self.train_epoch()

            if ddp.printable() and self.enable_checkpointing:
                self.train_eval()
            if self.world_size > 1:
                ddp.dist.barrier()
        
        if ddp.printable():
            for layer in self.concrete_model.module.bert.encoder.layer:
                layer = layer # type: sparse.BertLayer
                log(layer.p_logit.item(), layer.concrete_prop_p_logit.item())

# dispose

    def dispose(self):
        self.approx_trainer.dispose()

def main_ddp(rank, world_size, ddp_port, args):
    ddp.setup(rank, world_size, ddp_port)

    trainer = VitConcreteTrainer(
        subset=args.subset,
        model=args.model,
        factor=args.factor,
        batch_size=args.batch_size,
        device=rank,
        world_size=world_size,
        init_checkpoint=None,
        enable_valid=args.enable_valid,
        epochs=args.epochs,
        init_p_logit=args.p_logit,
        json_postfix=args.json_postfix,
    )
    trainer.main()

    ddp.cleanup()

def main(args):
    args.n_gpus = min(args.n_gpus, torch.cuda.device_count())
    ddp.spawn(main_ddp, (args,), args.n_gpus)

if  __name__ == '__main__':
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='base')
    parser.add_argument('--model', type=str, default='deit-small')
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--p-logit', type=float, default=-0.5)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--enable-valid', action='store_true', default=False)
    parser.add_argument('--json-postfix', type=str, default=None)

    args = parser.parse_args()
    log('given args', args)

    main(args)