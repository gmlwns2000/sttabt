"""
This script train
transformers.ViTForImageClassification
for specified dataset

***
"""

import random
import torch
from torch import optim, nn
import numpy as np
import tqdm
import transformers
from dataset.images_hf import ImagesHfDataset, ExamplesToBatchTransform, ViTInputTransform
from utils import ddp, env_vars

from utils import initialize_saves_dirs
initialize_saves_dirs()

class VitApproxTrainerMViT:
    def __init__(self,
        subset = 'in1k',
        batch_size = -1,
        device = 0,
        world_size = 1,
        skip_dataloader = False,
        factor=4,
        epochs=10,
    ):
        self.seed()
        
        if batch_size <= 0:
            batch_size = 16
        self.batch_size = batch_size

        base_lr = 1e-3
        self.lr = base_lr * (batch_size*world_size/512.0)
        print('VitTrainer: Configured LR', self.lr)

        self.weight_decay = 0
        self.amp_enable = True
        self.factor = factor
        
        self.model_id = 'mvit-tiny'
        self.base_model_id = 'mvit-tiny'
        self.base_model_id_hf = 'mvit-tiny'
        self.subset = subset
        assert subset == 'in1k'
        self.device = device
        self.epochs = epochs
        self.device = device
        self.world_size = world_size

        if not skip_dataloader:
            self.init_dataloader()
        self.reset_train()

# Initialize

    def init_dataloader(self, test_split=None):
        #dispose previous

        from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
        import timm

        assert self.model_id in ['deit-small', 'lvvit-small', 'mvit-tiny']

        data_dir = env_vars.get_imagenet_root()
        dataset_name = 'imagenet'
        arg_train_split = 'train'
        arg_val_split = 'validation'
        arg_prefetcher = True
        arg_pin_mem = False
        arg_nworker_train = 12
        arg_nworker_test = 8
        arg_distributed = self.world_size > 1

        model_id = 'deit_small_patch16_224'
        print(f"VitApproxTrainerMViT: timm dataloader of ({model_id}) b{self.batch_size}")
        model = timm.create_model(model_id, pretrained=True)

        data_config = resolve_data_config({}, model=model, verbose=ddp.printable())

        train_interpolation = 'random'
        if False or not train_interpolation:
            train_interpolation = data_config['interpolation']

        # create the train and eval datasets
        if test_split == 'split':
            print("VitApproxTrainerMViT: Using valid split from train set for test set.")
            dataset_train = create_dataset(
                dataset_name, root=data_dir, split=arg_train_split, is_training=True,
                #class_map=args.class_map,
                #download=args.dataset_download,
                batch_size=self.batch_size,
                repeats=self.epochs,
            )
            #data = dataset_train.shuffle(seed=42).train_test_split(test_size=0.05)
            # dataset_train = data['train']
            # dataset_eval = data['test']
            
            #eval_size = int(len(dataset_train) * 0.05)
            # dataset_train, dataset_eval = torch.utils.data.random_split(
            #     dataset_train, [len(dataset_train) - eval_size, eval_size]
            # )
            def train_val_dataset(dataset, val_split=0.05):
                # https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5
                #from torch.utils.data import Subset
                from torch.utils.data import Dataset
                class Subset(Dataset):
                    r"""
                    Subset of a dataset at specified indices.

                    Args:
                        dataset (Dataset): The whole Dataset
                        indices (sequence): Indices in the whole set selected for subset
                    """

                    def __init__(self, dataset, indices) -> None:
                        self.dataset = dataset
                        self.indices = indices

                    def __getitem__(self, idx):
                        ret = None
                        if isinstance(idx, list):
                            ret = self.dataset[[self.indices[i] for i in idx]]
                        else:
                            ret = self.dataset[self.indices[idx]]
                        img, target = ret
                        if self.transform is not None:
                            img = self.transform(img)
                        return img, target

                    def __len__(self):
                        return len(self.indices)

                from sklearn.model_selection import train_test_split
                train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

                datasets = {}
                
                datasets['train'] = Subset(dataset, train_idx)
                datasets['val'] = Subset(dataset, val_idx)
                # datasets['train'].__getitem__ == MethodType(__getitem__, datasets['train'])
                # datasets['val'].__getitem__ == MethodType(__getitem__, datasets['val'])
                return datasets
            ds = train_val_dataset(dataset_train)
            dataset_train = ds['train']
            dataset_eval = ds['val']
        else:
            dataset_train = create_dataset(
                dataset_name, root=data_dir, split=arg_train_split, is_training=True,
                #class_map=args.class_map,
                #download=args.dataset_download,
                batch_size=self.batch_size,
                repeats=self.epochs,
            )
            dataset_eval = create_dataset(
                dataset_name, root=data_dir, split=arg_val_split, is_training=False,
                #class_map=args.class_map,
                #download=args.dataset_download,
                batch_size=self.batch_size
            )

        collate_fn = None

        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=self.batch_size,
            is_training=True,
            use_prefetcher=arg_prefetcher,
            no_aug=False,
            re_prob=0.,
            re_mode='pixel',
            re_count=1,
            re_split=False,
            scale=[0.08, 1.0],
            ratio=[3./4., 4./3.],
            hflip=0.5,
            vflip=0.,
            color_jitter=0.4,
            auto_augment=None,
            num_aug_repeats=0,
            num_aug_splits=0,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=arg_nworker_train,
            distributed=arg_distributed,
            collate_fn=collate_fn,
            pin_memory=arg_pin_mem,
            use_multi_epochs_loader=False,
            worker_seeding='all',
        )

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=self.batch_size,
            is_training=False,
            use_prefetcher=arg_prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=arg_nworker_test,
            distributed=False,
            crop_pct=data_config['crop_pct'],
            pin_memory=arg_pin_mem,
        )

        self.timm_data_train = loader_train
        self.timm_data_test = loader_eval

    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def reset_train(self):
        self.seed()

        self.epoch = 0
        
        from models import mvit_timm
        self.main_model = mvit_timm.mvitv2_tiny_sttabt(pretrained=True)
        self.main_model = self.main_model.to(self.device)

        self.model = mvit_timm.init_approx_net_from(
            self.main_model, factor=self.factor
        )
        self.model = self.model.to(self.device)
        self.model = ddp.wrap_model(self.model, find_unused_paramters=True)

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
        return f'./saves/vit-approx-{self.base_model_id}-{self.subset}-f{self.factor}.pth'

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
        print('VitApproxTrainerMViT: Checkpoint saved', self.get_checkpoint_path())
    
    def load(self):
        state = torch.load(self.get_checkpoint_path(), map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        del state
        print('VitApproxTrainerMViT: Checkpoint loaded', self.get_checkpoint_path())

# Eval Impl

    def eval_model(self, model, show_message=False):
        model.eval()
        from datasets import load_metric

        metric = load_metric("accuracy")

        pbar = tqdm.tqdm(self.timm_data_test, desc='eval')
        for batch in pbar:
            # batch = {k: batch[k].to(self.device, non_blocking=True) for k in batch.keys()}
            batch = {'pixel_values': batch[0].to(self.device), 'labels': batch[1].to(self.device)} #timm compatibility
            labels = batch['labels']
            del batch['labels']

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enable):
                output = model(**batch)

            metric.add_batch(predictions=torch.argmax(output.logits, dim=-1), references=labels)
        score = metric.compute()
        if show_message: print(score)
        return score

# Train Impl

    def train_epoch(self):
        self.model.train()

        pbar = tqdm.tqdm(self.timm_data_train)
        for step, batch in enumerate(pbar):
            # if step > 100: break
            batch = {'pixel_values': batch[0].to(self.device), 'labels': batch[1].to(self.device)} #timm compatibility
            # batch = {k: batch[k].to(self.device, non_blocking=True) for k in batch.keys()}
            
            with torch.cuda.amp.autocast(enabled=self.amp_enable):
                output = self.model(**batch)
                loss = output['loss']
                loss_att = output['loss_details']['loss_att']
                loss_hid = output['loss_details']['loss_hid']
                loss_emb = output['loss_details']['loss_emb']
                loss_pred = output['loss_details']['loss_pred']
            
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            pbar.set_description(
                f'[{self.epoch+1}/{self.epochs}] {loss.item():.5f} '+\
                f'Latt:{loss_att.item():.5f} Lhid:{loss_hid.item():.5f} Lemb:{loss_emb.item():.5f} Lpred:{loss_pred.item():.5f} '
            )

# Main

    def main(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            
            if (self.world_size > 1 and ddp.printable()) or self.world_size <= 1:
                self.eval_model(self.model, show_message=True)
                self.save()
            
            if self.world_size > 1:
                ddp.barrier()

def main_ddp(rank, world_size, ddp_port, args):
    ddp.setup(rank, world_size, ddp_port)

    print('Worker:', rank, world_size, ddp_port, ddp.printable())
    trainer = VitApproxTrainerMViT(
        batch_size=args.batch_size,
        world_size=world_size,
        device=rank,
    )
    trainer.main()

    ddp.cleanup()

def main():
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--n-gpus', type=int, default=1)

    args = parser.parse_args()

    ddp.spawn(main_ddp, args=(args,), n_gpus=args.n_gpus)

if __name__ == '__main__':
    main()