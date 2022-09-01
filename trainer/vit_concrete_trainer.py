import gc
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

def log(*args):
    print("VitConcreteTrainer:", *args)

class VitApproxTrainer:
    def __init__(self,
        subset = 'base',
        model = 'deit-small',
        factor = 4,
        batch_size = -1,
        device = 0,
        world_size = 1,
        init_checkpoint=None,
        enable_valid = False,
    ):
        self.device = device
        if batch_size < 0:
            self.batch_size = vit_approx.tasks_to_batch_size[model]
        else:
            self.batch_size = batch_size
        self.factor = factor
        self.world_size = world_size
        self.model_id = model

        self.enable_valid = enable_valid
        self.lr = 1e-5
        self.weight_decay = 5e-2
        self.init_p_logit = -0.5
        self.epochs = 30

        self.approx_trainer = vit_approx.VitApproxTrainer(
            subset = subset,
            model = model,
            factor = factor,
            batch_size = batch_size,
            device = device,
            world_size = world_size,
            init_checkpoint = init_checkpoint,
            enable_valid = self.enable_valid,
        )
        #self.approx_trainer.init_dataloader("split" if self.enable_valid else "val")
        assert self.approx_trainer.dataloader_lib == 'timm'
        try:
            self.approx_trainer.load()
        except:
            log("Failed to load attention approximation!")
        
        self.init_concrete()
        
        self.init_optim()
    
    def init_concrete(self):
        self.concrete_model = sparse.ApproxSparseBertForSequenceClassification(
            self.approx_trainer.model.config,
            self.approx_trainer.approx_bert.module,
            arch = 'vit'
        )
        
        try:
            self.concrete_model.bert.load_state_dict(
                vit_approx.get_vit(self.approx_trainer.model),
                strict=False,
            )
        except Exception as ex:
            log('load vit', ex)
        
        try:
            self.concrete_model.classifier.load_state_dict(
                self.approx_trainer.model.classifier,
                strict=False,
            )
        except:
            log('load classifier', ex)
        
        self.concrete_model.to(self.device).train()
        self.concrete_model.use_concrete_masking = True
        self.concrete_model = ddp.wrap_model(self.concrete_model, find_unused_paramters=False)

        self.set_concrete_init_p_logit(self.init_p_logit)
        self.set_concrete_hard_threshold(None)
    
    def set_concrete_init_p_logit(self, v):
        self.concrete_model.module.bert.set_concrete_init_p_logit(v)
    
    def set_concrete_hard_threshold(self, v):
        self.concrete_model.module.bert.set_concrete_hard_threshold(v)
    
    def init_optim(self):
        pass

    def get_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        high_lr = ['p_logit']
        params = [
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in high_lr))], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay)) and (any(nd in n for nd in high_lr))], 'lr':self.lr * 10, 'weight_decay': 0},
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

    def eval_concrete(self):
        # check average loss
        sparse.benchmark_concrete_occupy(True)
        loss_sum = 0
        loss_count = 0
        self.concrete_model.eval()
        ddp_model = self.concrete_model
        self.concrete_model = ddp_model.module
        for i, batch in enumerate(self.test_dataloader):
            if i > 100: break
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled = False):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                batch['output_attentions'] = True
                loss = self.concrete_model(**batch).loss
            
            loss_sum += loss.item()
            loss_count += 1
        valid_loss = loss_sum / loss_count
        print('valid loss:', valid_loss)

        # check accuracy
        sparse.benchmark_reset()
        result = self.eval_sparse_model(show_message=False, max_step=100)
        est_k = sparse.benchmark_get_average('concrete_occupy')
        # print('concrete_occupy', est_k)
        # print('evaluate sparse net. score:', result)

        self.concrete_model = ddp_model
        self.concrete_model.train()

# train function

    def train_eval(self):
        pass

    def train_epoch(self):
        pass

    def main(self):
        for epoch in range(self.epochs):
            gc.collect()
            torch.cuda.empty_cache()
        
            if epoch >= min(self.epochs - 1, (self.epochs - 1) * 0.8):
                if self.enable_checkpointing: print('train hard prune')
                self.set_concrete_hard_threshold(0.5)

            self.train_epoch()

            if ddp.printable() and self.enable_checkpointing:
                self.eval_concrete(show_message=True)
                for layer in self.concrete_model.module.bert.encoder.layer:
                    layer = layer # type: sparse.BertLayer
                    print(layer.p_logit.item(), layer.concrete_prop_p_logit.item())

# dispose

    def dispose(self):
        self.approx_trainer.dispose()

if  __name__ == '__main__':
    pass