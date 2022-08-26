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

tasks_to_dataset = {
    'base': './dataset/imagenet_hf.py',
    'cifar100': 'cifar100',
    'imagenet': './dataset/imagenet_hf.py',
}

tasks_to_split = {
    'base': 'train',
    'cifar100': 'train',
    'imagenet': 'train',
}

tasks_to_test_split = {
    'base': 'split',
    'cifar100': 'test',
    'imagenet': 'val',
}

#pretrained model
model_to_hf = {
    'vit-base': 'google/vit-base-patch16-224-in21k',
    'deit-base': 'facebook/deit-base-patch16-224',
    'deit-small': 'facebook/deit-small-patch16-224',
    'deit-small-distilled': 'facebook/deit-small-distilled-patch16-224',
}

finetuned_to_hf = {
    'imagenet': {
        'vit-base': 'google/vit-base-patch16-224',
        'deit-base': 'facebook/deit-base-patch16-224',
        'deit-small': 'facebook/deit-small-patch16-224',
        'deit-small-distilled': 'facebook/deit-small-distilled-patch16-224',
    }
}

from utils.load_state_dict_interpolated import load_state_dict_interpolated

def get_vit(model):
    if hasattr(model, 'vit'):
        return model.vit
    elif hasattr(model, 'deit'):
        return model.deit
    else: 
        raise Exception('vit is not found')

class VitApproxTrainer:
    def __init__(self,
        subset = 'base',
        model = 'deit-base',
        factor = 4,
        batch_size = -1,
        device = 0,
        world_size = 1,
        init_checkpoint=None,
    ):
        self.seed()

        self.lr = 1e-4
        self.weight_decay = 0
        self.amp_enable = False

        self.model_id = model
        self.model_id_hf = model_to_hf[model]
        self.subset = subset
        self.device = device
        self.factor = factor
        self.epochs = tasks_to_epoch[self.subset]
        if batch_size <= 0:
            batch_size = tasks_to_batch_size[self.model_id]
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
        self.extractor = transformers.AutoFeatureExtractor.from_pretrained(self.model_id_hf)
        self.dataset = ImagesHfDataset(
            ExamplesToBatchTransform(ViTInputTransform(self.extractor)),
            ExamplesToBatchTransform(ViTInputTransform(self.extractor, test=True)),
            batch_size=self.batch_size,
            name=tasks_to_dataset[self.subset],
            split=tasks_to_split[self.subset],
            test_split=tasks_to_test_split[self.subset],
            num_workers_test=8,
            num_workers_train=8,
        )

    def get_base_model(self) -> "transformers.ViTForImageClassification":
        if self.model_id in ['vit-base', 'deit-base', 'deit-small']:
            model_cls = transformers.ViTForImageClassification
        elif self.model_id in ['deit-base-distilled', 'deit-small-distilled']:
            model_cls = transformers.DeiTForImageClassification
        else: raise Exception()

        if self.subset == 'base':
            return model_cls.from_pretrained(self.model_id_hf)
        elif self.subset in ['cifar100']:
            #trained from trainer.vit_trainer
            base_model = model_cls.from_pretrained(
                self.model_id_hf,
                num_labels=self.dataset.num_labels,
                id2label=self.dataset.id2label,
                label2id=self.dataset.label2id
            )
            assert self.model_id in ['vit-base', 'deit-base', 'deit-small']
            state = torch.load(f'./saves/{self.model_id}-{self.subset}.pth', map_location='cpu')
            base_model.load_state_dict(state['model'])
            del state
            return base_model
        elif self.subset in ['imagenet']:
            #trained from huggingface models
            base_model_hf = finetuned_to_hf[self.subset][self.model_id]
            base_model = model_cls.from_pretrained(base_model_hf)
            return base_model
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
        load_state_dict_interpolated(self.approx_bert.bert, get_vit(self.model).state_dict())
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
    
    def set_batch_size(self, v):
        self.dataset.batch_size = self.batch_size = v

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
        #self.model.load_state_dict(state['model'])
        try:
            self.approx_bert.load_state_dict(state['approx_bert'])
        except Exception as ex:
            print('VitApproxTrainer: Error while loading approx bert')
            print(ex)
        if 'subset' in state and state['subset'] == self.subset:
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

    def eval_model_metric(self, model, show_message=False):
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
    
    def eval_model_metric_base(self, show_message=False):
        return self.eval_model_metric(self.model, show_message=show_message)

    def eval_model_metric_approx(self, show_message=False):
        return self.eval_model_metric(self.approx_bert, show_message=show_message)
    
    def eval_model_metric_sparse(self, ks=0.5, mode='sparse', show_message=False):
        sparse_bert = sparse.SparseBertModel(self.model.config, add_pooling_layer=False, arch='vit')
        try:
            report = sparse_bert.load_state_dict(get_vit(self.model).state_dict(), strict=False)
            ignored_term = ['p_logit', 'ltp']
            missing_keys = [k for k in report.missing_keys if not any(n in k for n in ignored_term)]
            unexpected_keys = [k for k in report.unexpected_keys if not any(n in k for n in ignored_term)]
            if len(missing_keys) > 0:
                print('VitApproxTrainer.eval_model_metric_sparse: missing_keys', missing_keys)
            if len(unexpected_keys) > 0:
                print('VitApproxTrainer.eval_model_metric_sparse: unexpected_keys', unexpected_keys)
        except Exception as ex:
            print('Error while create sparse bert', ex)
        wrapped_bert = sparse.ApproxSparseBertModel(sparse_bert=sparse_bert, approx_bert=self.approx_bert.module, ks=ks, arch='vit')
        occupy_metric = None
        if mode == 'forward':
            wrapped_bert.use_forward_sparse = True
            wrapped_bert.run_original_attention = False
            occupy_metric = 'forward_occupy'
        elif mode == 'absatt':
            wrapped_bert.use_forward_sparse = False
            wrapped_bert.run_original_attention = True
            occupy_metric = 'mask_occupy'
        elif mode == 'sparse':
            wrapped_bert.use_forward_sparse = False
            wrapped_bert.run_original_attention = False
            occupy_metric = 'mask_occupy'
        else:
            raise Exception('unknown mode')
        
        sparse_cls_vit = transformers.ViTForImageClassification(self.model.config)
        sparse_cls_vit.load_state_dict(self.model.state_dict())
        sparse_cls_vit.vit = wrapped_bert
        sparse_cls_vit.to(self.device).eval()
        sparse.benchmark_reset()
        metric = self.eval_model_metric(sparse_cls_vit, show_message=show_message)
        occupy = sparse.benchmark_get_average(occupy_metric)
        metric['occupy'] = occupy

        return metric

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
                original_emb = get_vit(self.model).embeddings(batch['pixel_values'])
            
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
        
        if self.subset != 'base':
            metric = self.eval_model_metric_approx()
            for k in metric.keys():
                loss_sum[k] = metric[k]
        
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
            batch = {k: torch.tensor(batch[k]).to(self.device, non_blocking=True) for k in batch.keys()}
            if self.world_size != 1:
                batch = {
                    k: v[self.device*(self.batch_size//self.world_size):(self.device+1)*(self.batch_size//self.world_size)] 
                    for k, v in batch.items()
                }
            batch['output_attentions'] = True
            batch['output_hidden_states'] = True
            if 'labels' in batch: del batch['labels']

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enable):
                original_output = self.model(**batch)
                original_emb = get_vit(self.model).embeddings(batch['pixel_values'])
            
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
        if self.subset != 'base':
            print('Evaluate loaded base model')
            self.eval_model_metric_base(show_message=True)
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            ddp.barrier()
            if ddp.printable():
                self.eval_model(self.model, self.approx_bert.module, show_message=True)
                self.save()
            ddp.barrier()
        
        self.dispose()
    
    def main_eval(self):
        self.load()
        batch_size = self.batch_size
        metric_baseline = self.eval_model_metric_base()
        print('baseline', metric_baseline)
        metric_distil = self.eval_model_metric_approx()
        print('distil', metric_distil)
        target_ks = 0.45
        if target_ks <= 0.666:
            ksx = [target_ks*0.5+((1-x/10.0)**1.0) * target_ks for x in range(12)]
        else:
            ksx = [(1-x/10.0)*(2-2*target_ks)+(2*target_ks-1) for x in range(12)]
        metric_forward = self.eval_model_metric_sparse(ks=ksx, mode='forward')
        print('forward', metric_forward)
        self.dataset.batch_size = self.batch_size = 1
        metric_sparse = self.eval_model_metric_sparse(ks=0.25, mode='sparse')
        print('forward', metric_sparse)
        metric_absatt = self.eval_model_metric_sparse(ks=0.25, mode='absatt')
        print('forward', metric_absatt)
        self.dataset.batch_size = self.batch_size = batch_size

        self.dispose()
    
    def dispose(self):
        self.dataset.dispose()

def main_ddp(rank, world_size, ddp_port, args):
    ddp.setup(rank, world_size, ddp_port)

    print('Worker:', rank, world_size, ddp_port, ddp.printable())
    main_trainer(rank, world_size, args)

    ddp.cleanup()

def main_trainer(device, world_size, args):
    trainer = VitApproxTrainer(
        device=device,
        world_size=world_size,
        batch_size=args.batch_size,
        subset=args.subset,
        factor=args.factor,
        init_checkpoint=args.init_checkpoint,
        model=args.model,
    )
    if not args.eval:
        trainer.main()
    else:
        trainer.main_eval()

def main():
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='base')
    parser.add_argument('--model', type=str, default='deit-base')
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--init-checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--eval', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    if not args.eval:
        ddp.spawn(main_ddp, args=(args,))
    else:
        main_trainer(0, 1, args)

if __name__ == '__main__':
    main()