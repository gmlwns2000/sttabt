import transformers, torch, tqdm, random, gc, visdom
import numpy as np
import transformers.models.bert.modeling_bert as berts
import models.sparse_token as sparse
from datasets import load_dataset, load_metric
from datasets.utils import logging as datasets_logging
from torch import optim, nn
from utils import ThreadBuffer
from utils.gpu_pool import print
import torch.nn.functional as F

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import initialize_saves_dirs
initialize_saves_dirs()

def setup(rank, world_size, port=32277):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

datasets_logging.set_verbosity_error()

torch.cuda.empty_cache()

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_epochs = {
    "cola": 10,
    "mnli": 2,
    "mrpc": 10,
    "qnli": 2,
    "qqp":  2,
    "rte":  10,
    "sst2": 2,
    "stsb": 10,
    "wnli": 10,
    "bert": 6,
}

#My gpu is so tiny and cute <3, so I had to reduce batch size.
#After my few month of shoveling, I found out I have to use batch size 64 (from LTP github).
#So I add gradient accum.
task_to_batch_size = {
    "cola": 32,
    "mnli": 4,
    "mrpc": 32,
    "qnli": 4,
    "qqp":  8,
    "rte":  8,
    "sst2": 16,
    "stsb": 16,
    "wnli": 32,
    "bert": 8,
}

task_to_grad_accum_step = {
    "cola": 2,
    "mnli": 16,
    "mrpc": 2,
    "qnli": 16,
    "qqp":  8,
    "rte":  8,
    "sst2": 4,
    "stsb": 4,
    "wnli": 2,
}

def get_dataloader(subset, tokenizer, batch_size, split='train'):
    if subset == 'bert':
        subset = "cola" #return dummy set
    
    dataset = load_dataset('glue', subset, split=split, cache_dir='./cache/datasets')
    
    sentence1_key, sentence2_key = task_to_keys[subset]

    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, max_length=512, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, batch_size=1024)
    if split.startswith('train[:'):
        dataset = dataset.sort('label')
        dataset = dataset.shuffle(seed=random.randint(0, 10000))
    dataset = dataset.map(encode, batched=True, batch_size=1024)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def get_base_model(dataset, only_tokenizer=False):
    checkpoint = {
        "cola": "textattack/bert-base-uncased-CoLA",
        "mnli": "yoshitomo-matsubara/bert-base-uncased-mnli",
        "mrpc": "textattack/bert-base-uncased-MRPC",
        "qnli": "textattack/bert-base-uncased-QNLI",
        "qqp": "textattack/bert-base-uncased-QQP",
        "rte": "textattack/bert-base-uncased-RTE",
        "sst2": "textattack/bert-base-uncased-SST-2",
        "stsb": "textattack/bert-base-uncased-STS-B",
        "wnli": "textattack/bert-base-uncased-WNLI",
        "bert": "bert-base-uncased",
    }[dataset]

    model = {
        "cola": berts.BertForSequenceClassification,
        "mnli": berts.BertForSequenceClassification,
        "mrpc": berts.BertForSequenceClassification,
        "qnli": berts.BertForSequenceClassification,
        "qqp": berts.BertForSequenceClassification,
        "rte": berts.BertForSequenceClassification,
        "sst2": berts.BertForSequenceClassification,
        "stsb": berts.BertForSequenceClassification,
        "wnli": berts.BertForSequenceClassification,
        "bert": berts.BertForSequenceClassification,
    }[dataset]
    
    tokenizer = transformers.BertTokenizerFast.from_pretrained(checkpoint)
    if only_tokenizer:
        return None, tokenizer
    
    bert = model.from_pretrained(checkpoint, cache_dir='./cache/huggingface/')
    return bert, tokenizer

class MimicDDP(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class LtpTrainer:
    def __init__(self, 
        dataset,
        batch_size=None, device=0, world_size=1, 
        checkpoint_name=None, init_checkpoint=None,
        ltp_lambda=None, ltp_temperature=None,
        enable_plot=False,
    ):
        print('Trainer:', dataset)
        self.seed()
        
        self.enable_checkpointing = True
        self.enable_plot = enable_plot
        self.init_checkpoint = init_checkpoint
        self.checkpoint_name = checkpoint_name
        self.ltp_lambda = ltp_lambda
        self.ltp_temperature = ltp_temperature
        self.lr = 2e-5 #from LTP github
        # if dataset == 'mnli':
        #     print('LtpTrainer: MNLI LR 1e-6')
        #     self.lr = 1e-6
        self.weight_decay = 0
        self.dataset = dataset
        if batch_size is None or batch_size <= 0:
            batch_size = task_to_batch_size[self.dataset]
        self.batch_size = batch_size
        self.device = device
        self.world_size = world_size
        self.epoch = 0
        self.gradient_accumulate_steps = task_to_grad_accum_step[self.dataset]

        _, self.tokenizer = get_base_model(self.dataset, only_tokenizer=True)

        if device == 0:
            self.train_dataloader = get_dataloader(
                self.dataset, self.tokenizer, self.batch_size)
        if self.world_size > 1:
            dist.barrier()

        _batch_size = self.batch_size
        self.batch_size = None
        self.set_batch_size(_batch_size)

        self.epochs = task_to_epochs[self.dataset]
        
        self.model, self.tokenizer = get_base_model(self.dataset)
        self.model.eval()
        self.model.to(self.device)
        self.model_bert = self.model.bert
        self.model_classifier = self.model.classifier
        
        self.init_sparse_bert()
        
        self.optimizer = self.get_optimizer(self.sparse_bert)
        self.scaler = torch.cuda.amp.GradScaler()

        self.last_metric_score = None
        self.last_loss = None

        print('Trainer: Checkpoint path', self.checkpoint_path())

        if not (self.init_checkpoint is None):
            print('Trainer: From pretrained checkpoint', self.init_checkpoint)
            self.load(self.init_checkpoint)

        self.tqdm_position = 0
    
    def init_sparse_bert(self):
        self.sparse_bert_inner = sparse.SparseBertForSequenceClassification(self.model.config)
        self.sparse_bert_inner.load_state_dict(self.model.state_dict(), strict=False)
        self.sparse_bert_inner.to(self.device).train()
        self.sparse_bert_inner.bert.set_ltp_prune_token(True)
        self.sparse_bert_inner.bert.set_ltp_prune_token_soft_pruning(True)
        if self.ltp_temperature is not None:
            self.sparse_bert_inner.bert.set_ltp_temperature(self.ltp_temperature)
        if self.ltp_lambda is not None:
            self.sparse_bert_inner.ltp_lambda = self.ltp_lambda
        
        if self.world_size > 1:
            self.sparse_bert = DDP(self.sparse_bert_inner, device_ids=[self.device], find_unused_parameters=False)
        else:
            self.sparse_bert = MimicDDP(self.sparse_bert_inner)
    
    def reset_train(self):
        self.seed()

        self.epoch = 0
        self.last_metric_score = None
        self.last_loss = None
        
        self.init_sparse_bert()

        self.optimizer = self.get_optimizer(self.sparse_bert)
        self.scaler = torch.cuda.amp.GradScaler()

    def load_train_dataset(self):
        self.train_dataloader = get_dataloader(
            self.dataset, self.tokenizer, self.batch_size, split='train[:90%]')

    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
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

    def set_batch_size(self, new_value):
        if new_value != self.batch_size:
            print("GlueAttentionApproxTrainer: update batch size", new_value)
            self.batch_size = new_value
            
            self.load_train_dataset()

            self.valid_dataloader = get_dataloader(
                self.dataset, self.tokenizer, self.batch_size, split='train[-10%:]')

            split = {
                "cola": "validation",
                "mnli": "validation_matched",
                "mrpc": "test",
                "qnli": "validation",
                "qqp": "validation",
                "rte": "validation",
                "sst2": "validation",
                "stsb": "validation",
                "wnli": "validation",
                "bert": "validation",
            }[self.dataset]
            self.test_dataloader = get_dataloader(
                self.dataset, self.tokenizer, self.batch_size, split=split)
            self.epochs = task_to_epochs[self.dataset]

# checkpoint functions

    def checkpoint_path(self):
        if self.checkpoint_name is not None:
            return f'saves/{self.checkpoint_name}.pth'
        return f'saves/ltp-glue-{self.dataset}.pth'
    
    def load(self, path = None, load_loss = True):
        if path is None:
            path = self.checkpoint_path()
        else:
            load_loss = False
        state = torch.load(path, map_location='cpu')
        #self.model.load_state_dict(state['bert'])
        try:
            self.sparse_bert.load_state_dict(state['sparse_bert'], strict=False)
        except RuntimeError as ex:
            print("Trainer: Error during state dict load")
            print(ex)
        if load_loss:
            if 'last_metric_score' in state: self.last_metric_score = state['last_metric_score']
            if 'last_loss' in state: self.last_loss = state['last_loss']
        print('loaded', state['epochs'], state['last_loss'])
        del state

    def save(self):
        torch.save({
            'bert':self.model.state_dict(),
            'sparse_bert': self.sparse_bert.state_dict(),
            'trained_epoch': self.epoch,
            'epochs':self.epochs,
            'last_metric_score':self.last_metric_score,
            'last_loss':self.last_loss,
        }, self.checkpoint_path())
        print('saved', self.checkpoint_path())

# eval functions

    def eval_base_model(self, model = None, amp = False, show_messages=True, max_step=987654321, split='test'):
        self.seed()
        if model is None:
            model = self.model
        model.eval()
        
        if self.dataset == 'bert':
            metric = load_metric('glue', 'cola')
        else:
            metric = load_metric('glue', self.dataset)
        avg_length = 0
        step_count = 0
        
        if split == 'test':
            dataloader = self.test_dataloader
        elif split == 'valid':
            dataloader = self.valid_dataloader
        else:
            raise Exception()
        
        loss_sum = 0
        loss_count = 0
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc='eval', position=self.tqdm_position)):
            if i > max_step: break
            step_count += 1

            batch = {k: v.to(self.device) for k, v in batch.items()}
            #print(batch['attention_mask'].shape, torch.mean(torch.sum(batch['attention_mask'], dim=-1).float()).item())
            avg_length += torch.mean(torch.sum(batch['attention_mask'], dim=-1).float()).item() / batch['attention_mask'].shape[-1]
            labels = batch['labels']
            #del batch['labels']
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                outputs = model(**batch)
            loss = outputs[0]
            loss_sum += loss.item()
            loss_count += 1
            predictions = outputs[1]

            if self.dataset != 'stsb': 
                predictions = torch.argmax(predictions, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        
        score = metric.compute()
        self.last_metric_score = score
        if show_messages:
            print('metric score', score)
            print('avg occupy', avg_length / step_count)
        gc.collect()
        torch.cuda.empty_cache()
        score['loss'] = loss_sum / loss_count
        return score

    def eval_sparse_model(self,
        show_message=True,
        max_step=987654321,
        split='test'
    ):
        self.seed()
        sparse_result = self.eval_base_model(model = self.sparse_bert, show_messages = show_message, max_step=max_step, split=split)
        return sparse_result

    def eval_main(self, ks='dynamic', split='test'):
        self.load()
        
        bert_result = self.eval_base_model(model = self.model, split=split)
        sparse_result = self.eval_sparse_model()
        est_k = sparse.benchmark_get_average('ltp_occupy')
        print('ltp_occupy', est_k)

        print(bert_result, sparse_result)

# train functions
    def prepare_plots(self):
        pass
    
    def train_plot(self,
        loss, loss_att, loss_hid, loss_emb, loss_pred
    ):
        pass

    def train_epoch(self):
        pbar = self.train_dataloader
        
        print_log = self.device == 0 or self.world_size == 1
        if print_log:
            pbar = tqdm.tqdm(pbar, position = self.tqdm_position)
        
        if self.epoch > self.epochs * 0.5 - 0.1:
            self.sparse_bert.module.bert.set_ltp_prune_token_soft_pruning(False)
        self.sparse_bert.train()

        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            if self.world_size > 1:
                batch = {k: v[self.device*(self.batch_size//self.world_size):(self.device+1)*(self.batch_size//self.world_size)] for k, v in batch.items()}
            batch['output_attentions'] = True
            batch['output_hidden_states'] = True
            #if 'labels' in batch: del batch['labels']
            
            with torch.cuda.amp.autocast():
                output = self.sparse_bert(**batch)
                loss = output.loss
            
            self.scaler.scale(loss/self.gradient_accumulate_steps).backward()

            if ((step+1) % self.gradient_accumulate_steps) == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.sparse_bert.parameters(), 0.5)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.last_loss = loss.item()
            if print_log:
                pbar.set_description(f"[{self.epoch+1}/{self.epochs}] L:{loss:.5f}")
            self.steps += 1

    def train_validate(self):
        # check average loss
        loss_sum = 0
        loss_count = 0
        self.sparse_bert.eval()
        ddp_model = self.sparse_bert
        self.sparse_bert = ddp_model.module
        for i, batch in enumerate(self.test_dataloader):
            if i > 100: break
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                batch['output_attentions'] = True
                loss = self.sparse_bert(**batch).loss
            
            loss_sum += loss.item()
            loss_count += 1
        valid_loss = loss_sum / loss_count
        print('valid loss:', valid_loss)

        # check accuracy
        sparse.benchmark_reset()
        result = self.eval_sparse_model(show_message=False, max_step=100)
        est_k = sparse.benchmark_get_average('ltp_occupy')
        print('ltp_occupy', est_k)
        print('evaluate sparse net. score:', result)

        self.sparse_bert = ddp_model
        self.sparse_bert.train()

        # save checkpoint with early stopping
        if self.best_test_loss >= valid_loss:
            if self.dataset != 'bert':
                self.best_test_loss = valid_loss # always save
            if self.enable_checkpointing: self.save()

    def main(self):
        self.best_test_loss = 987654321
        self.steps = 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
            
            self.train_epoch()

            self.load_train_dataset()

            if (self.device == 0 or self.world_size == 1) and self.enable_checkpointing:
                self.train_validate()
            if self.world_size > 1:
                dist.barrier()

def main_ddp(rank, world_size, args):
    print(f"Running DDP instance on rank {rank}:{args.port}.")
    setup(rank, world_size, args.port)

    trainer = LtpTrainer(
        args.subset,
        batch_size=args.batch_size,
        device=rank,
        world_size=world_size,
        checkpoint_name=args.checkpoint_name,
        init_checkpoint=args.init_checkpoint,
        enable_plot=args.enable_plot,
        ltp_lambda=args.ltp_lambda,
        ltp_temperature=args.ltp_temperature,
    )
    
    trainer.main()
    
    cleanup()

def main_eval(args):
    trainer = LtpTrainer(
        args.subset,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        init_checkpoint=args.init_checkpoint
    )
    trainer.eval_main()

def main(args):
    if args.eval:
        main_eval(args)
        return
    
    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    print(f'Setup DDP', n_gpus)
    if n_gpus == 1:
        main_ddp(0, n_gpus, args)
    else:
        mp.spawn(main_ddp,
            args=(n_gpus, args,),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--init-checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n-gpus', type=int, default=128)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--checkpoint-name', type=str, default=None)
    parser.add_argument('--enable-plot', action='store_true', default=False)
    parser.add_argument('--ltp-lambda', type=float, default=None)
    parser.add_argument('--ltp-temperature', type=float, default=None)

    args = parser.parse_args()
    if args.port < 0:
        args.port = random.randint(32000, 37000)
    #ngpus = torch.cuda.device_count()
    #args.device = args.device % ngpus
    print(args)
    
    main(args)
