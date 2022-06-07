import transformers, torch, tqdm, random, gc, visdom
import numpy as np
import transformers.models.bert.modeling_bert as berts
import models.sparse_token as sparse
from datasets import load_dataset, load_metric
from datasets.utils import logging as datasets_logging
from torch import optim, nn
from utils import ThreadBuffer
from dataset.wikitext import FilteredWikitext, WikitextBatchLoader
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '32277'

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
    "cola": 100,
    "mnli": 4,
    "mrpc": 200,
    "qnli": 20,
    "qqp":  4,
    "rte":  300,
    "sst2": 15,
    "stsb": 250,
    "wnli": 200,
    "bert": 200,
}

task_to_batch_size = {
    "cola": 64,
    "mnli": 8,
    "mrpc": 32,
    "qnli": 8,
    "qqp":  16,
    "rte":  8,
    "sst2": 16,
    "stsb": 16,
    "wnli": 32,
    "bert": 8,
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
    if split == 'train':
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

class GlueAttentionApproxTrainer:
    def __init__(self, dataset, factor, batch_size=None, device=0, world_size=1, wiki_train=False, wiki_epochs=5, checkpoint_name=None, init_checkpoint=None):
        print('Trainer:', dataset)
        self.seed()
        
        self.init_checkpoint = init_checkpoint
        self.wiki_train = wiki_train
        self.wiki_epochs = wiki_epochs
        self.checkpoint_name = checkpoint_name
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.factor = factor
        self.dataset = dataset
        if batch_size is None or batch_size <= 0:
            batch_size = task_to_batch_size[self.dataset]
        self.batch_size = batch_size
        self.device = device
        self.world_size = world_size

        _, self.tokenizer = get_base_model(self.dataset, only_tokenizer=True)

        if device == 0:
            self.train_dataloader = get_dataloader(
                self.dataset, self.tokenizer, self.batch_size)
        if self.world_size > 1:
            dist.barrier()

        self.load_train_dataset()

        self.split = {
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
            self.dataset, self.tokenizer, self.batch_size//self.world_size, split=self.split)
        self.epochs = task_to_epochs[self.dataset]
        if wiki_train:
            self.wiki_dataset = WikitextBatchLoader(batch_size=self.batch_size)
            self.lr = 1e-4 #2e-5
            self.weight_decay = 1e-4
            self.epochs = wiki_epochs
        
        self.model, self.tokenizer = get_base_model(self.dataset)
        self.model.eval()
        self.model.to(self.device)
        self.model_bert = self.model.bert
        self.model_classifier = self.model.classifier

        self.approx_bert = sparse.ApproxBertModel(self.model.config, factor=factor, wiki_train=wiki_train)
        self.approx_bert.train()
        self.approx_bert.to(self.device)
        if self.world_size > 1:
            self.approx_bert = DDP(self.approx_bert, device_ids=[device], find_unused_parameters=False)
        else:
            self.approx_bert = MimicDDP(self.approx_bert)
        self.optimizer = self.get_optimizer(self.approx_bert)
        self.scaler = torch.cuda.amp.GradScaler()

        self.last_metric_score = None
        self.last_loss = None

        print('Trainer: Checkpoint path', self.checkpoint_path())

        self.vis = visdom.Visdom()

        if not (self.init_checkpoint is None):
            print('Trainer: From pretrained checkpoint', self.init_checkpoint)
            self.load(self.init_checkpoint)
    
    def load_train_dataset(self):
        self.train_dataloader = get_dataloader(
            self.dataset, self.tokenizer, self.batch_size)

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

            self.train_dataloader = get_dataloader(
                self.dataset, self.tokenizer, self.batch_size)
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
            if self.wiki_train:
                self.wiki_dataset = WikitextBatchLoader(batch_size=self.batch_size)
                # self.lr = 2e-5
                # self.weight_decay = 5e-4
                self.epochs = self.wiki_epochs

# checkpoint functions

    def checkpoint_path(self):
        if self.checkpoint_name is not None:
            return f'saves/{self.checkpoint_name}.pth'
        if self.wiki_train: 
            if self.wiki_epochs == 3:
                return f'saves/glue-{self.dataset}-{self.factor}-wiki.pth'
            return f'saves/glue-{self.dataset}-{self.factor}-wiki-b{self.wiki_epochs}.pth'
        return f'saves/glue-{self.dataset}-{self.factor}.pth'
    
    def load(self, path = None, load_loss = True):
        if path is None:
            path = self.checkpoint_path()
        else:
            load_loss = False
        state = torch.load(path, map_location='cpu')
        #self.model.load_state_dict(state['bert'])
        try:
            self.approx_bert.load_state_dict(state['approx_bert'], strict=False)
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
            'approx_bert': self.approx_bert.state_dict(),
            'epochs':self.epochs,
            'last_metric_score':self.last_metric_score,
            'last_loss':self.last_loss,
        }, self.checkpoint_path())
        print('saved', self.checkpoint_path())

# eval functions

    def eval_base_model(self, model = None, amp = False, show_messages=True, max_step=987654321):
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
        
        for i, batch in enumerate(tqdm.tqdm(self.test_dataloader, desc='eval')):
            if i > max_step: break
            step_count += 1

            batch = {k: v.to(self.device) for k, v in batch.items()}
            #print(batch['attention_mask'].shape, torch.mean(torch.sum(batch['attention_mask'], dim=-1).float()).item())
            avg_length += torch.mean(torch.sum(batch['attention_mask'], dim=-1).float()).item() / batch['attention_mask'].shape[-1]
            labels = batch['labels']
            del batch['labels']
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                outputs = model(**batch)
            predictions = outputs[0]

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
        return score

    def eval_sparse_model(self, 
        ks=0.5, 
        use_forward=False,
        run_original_attention = False,
        show_message=True
    ):
        self.seed()
        wrapped_bert = sparse.ApproxSparseBertModel(self.model_bert, approx_bert=self.approx_bert, ks=ks)
        wrapped_bert.use_forward_sparse = use_forward
        wrapped_bert.run_original_attention = run_original_attention
        sparse_cls_bert = berts.BertForSequenceClassification(self.model_bert.config)
        sparse_cls_bert.load_state_dict(self.model.state_dict())
        sparse_cls_bert.bert = wrapped_bert
        sparse_cls_bert.to(self.device).eval()
        
        sparse_result = self.eval_base_model(model = sparse_cls_bert, show_messages = show_message)
        return sparse_result

    def eval_main(self, ks='dynamic'):
        self.load()
        
        bert_result = self.eval_base_model(model = self.model)
        sparse_result = self.eval_sparse_model(ks = ks)
        est_k = sparse.benchmark_get_average('est_k')
        print('est_k', est_k)

        print(bert_result, sparse_result)
    
    def eval_approx_model(
        self, show_message=True, max_step=987654321
    ):
        self.seed()
        approx_bert = self.approx_bert
        result = self.eval_base_model(model = approx_bert, show_messages = show_message, max_step = max_step)
        return result

# train functions
    def prepare_plots(self):
        win = self.vis.line(
            X=np.array([]),
            Y=np.array([]),
            win="loss",
            name='loss',
        )
    
    def train_plot(self,
        loss, loss_att, loss_hid, loss_emb, loss_pred
    ):
        self.vis.line(
            X=np.array([self.steps]),
            Y=np.array([loss.item()]),
            win="loss",
            name='loss',
            update='append',
        )

    def train_epoch(self):
        if self.wiki_train:
            pbar = self.wiki_dataset.__iter__()
        else:
            pbar = self.train_dataloader
        
        print_log = self.device == 0 or self.world_size == 1
        if print_log:
            pbar = tqdm.tqdm(pbar)
        
        self.model.eval()
        self.approx_bert.train()

        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            batch = {k: v[self.device*(self.batch_size//self.world_size):(self.device+1)*(self.batch_size//self.world_size)] for k, v in batch.items()}
            batch['output_attentions'] = True
            batch['output_hidden_states'] = True
            if 'labels' in batch: del batch['labels']
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                original_output = self.model(**batch)
                original_emb = self.model.bert.embeddings(batch['input_ids'])
            
            with torch.cuda.amp.autocast():
                batch['return_loss'] = True
                batch['original_output'] = original_output
                batch['original_emb'] = original_emb
                #tt = batch['input_ids'][0,1].item()
                #print(f'[S:{step}|D:{self.device}|{tt}]')
                approx_output, losses = self.approx_bert(**batch)
                loss, loss_att, loss_hid, loss_emb, loss_pred = losses
                if print_log and (self.steps % 5) == 0 and (self.dataset != 'bert'):
                    self.train_plot(loss, loss_att, loss_hid, loss_emb, loss_pred)

            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.approx_bert.parameters(), 0.5)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            self.last_loss = loss.item()
            if print_log:
                pbar.set_description(f"[{self.epoch+1}/{self.epochs}] L:{loss:.5f}, Latt:{loss_att:.5f}, Lhid:{loss_hid:.4f}, Lemb:{loss_emb:.4f}, Lpred:{loss_pred:.4f}")
            self.steps += 1

    def train_validate(self):
        # check average loss
        loss_sum = 0
        loss_count = 0
        self.approx_bert.eval()
        ddp_model = self.approx_bert
        self.approx_bert = ddp_model.module
        self.model.eval()
        for i, batch in enumerate(self.test_dataloader):
            if i > 100: break
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                batch['output_attentions'] = True
                if 'labels' in batch: del batch['labels']
                attentions = self.model(**batch).attentions

                approx_attentions = self.approx_bert(**batch).attentions
                loss = 0
                for j in range(len(attentions)):
                    loss += torch.mean(torch.square(approx_attentions[j]- attentions[j]))
                loss /= len(attentions)
            
            loss_sum += loss.item()
            loss_count += 1
        valid_loss = loss_sum / loss_count
        print('valid loss:', valid_loss)

        # check accuracy
        result = self.eval_approx_model(show_message=False, max_step=100)
        print('evaluate approx net. score:', result)

        self.approx_bert = ddp_model
        self.approx_bert.train()

        # save checkpoint with early stopping
        if self.best_test_loss >= valid_loss:
            if self.dataset != 'bert':
                self.best_test_loss = valid_loss # always save
            self.save()

    def main(self):
        self.best_test_loss = 987654321
        self.steps = 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
            
            self.train_epoch()

            self.load_train_dataset()

            if self.device == 0 or self.world_size == 1:
                self.train_validate()
            if self.world_size > 1:
                dist.barrier()

def main_ddp(rank, world_size, args):
    print(f"Running DDP instance on rank {rank}.")
    setup(rank, world_size)

    trainer = GlueAttentionApproxTrainer(
        args.subset,
        factor=args.factor,
        batch_size=args.batch_size,
        device=rank,
        world_size=world_size,
        wiki_train=args.wiki,
        wiki_epochs=args.wiki_epochs,
        checkpoint_name=args.checkpoint_name,
        init_checkpoint=args.init_checkpoint
    )
    
    trainer.main()
    
    cleanup()

def main_eval(args):
    trainer = GlueAttentionApproxTrainer(
        args.subset,
        factor=args.factor,
        batch_size=args.batch_size,
        device=args.device,
        wiki_train=args.wiki,
        wiki_epochs=args.wiki_epochs,
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
    mp.spawn(main_ddp,
        args=(n_gpus, args,),
        nprocs=n_gpus,
        join=True
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--init-checkpoint', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--factor', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n-gpus', type=int, default=128)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--not-wiki', action='store_true', default=False)
    parser.add_argument('--wiki-epochs', type=int, default=5)
    parser.add_argument('--checkpoint-name', type=str, default=None)

    args = parser.parse_args()
    args.wiki = not args.not_wiki
    #ngpus = torch.cuda.device_count()
    #args.device = args.device % ngpus
    print(args)
    
    main(args)
