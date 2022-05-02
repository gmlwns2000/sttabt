from threading import Thread
import transformers, torch, tqdm, random
import numpy as np
import transformers.models.bert.modeling_bert as berts
import models.sparse_token as sparse
from datasets import load_dataset, load_metric
from torch import optim, nn
from utils import ThreadBuffer
from dataset.wikitext import FilteredWikitext, WikitextBatchLoader

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
    "cola": 30,
    "mnli": 4,
    "mrpc": 30,
    "qnli": 20,
    "qqp":  4,
    "rte":  30,
    "sst2": 10,
    "stsb": 30,
    "wnli": 30,
}

task_to_batch_size = {
    "cola": 8,
    "mnli": 8,
    "mrpc": 8,
    "qnli": 4,
    "qqp":  8,
    "rte":  8,
    "sst2": 8,
    "stsb": 8,
    "wnli": 8,
}

def get_dataloader(subset, tokenizer, batch_size, split='train'):
    dataset = load_dataset('glue', subset, split=split, cache_dir='./cache/datasets/')
    
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
    
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset = dataset.map(encode, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def get_base_model(dataset):
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
    }[dataset]
    
    bert = model.from_pretrained(checkpoint, cache_dir='./cache/huggingface/')
    tokenizer = transformers.BertTokenizerFast.from_pretrained(checkpoint)
    
    return bert, tokenizer

class GlueAttentionApproxTrainer:
    def __init__(self, dataset, factor, batch_size=None, device=0, wiki_train=False, wiki_epochs=5):
        print('Trainer:', dataset)
        self.seed()
        
        self.wiki_train = wiki_train
        self.wiki_epochs = wiki_epochs
        self.lr = 5e-5
        self.weight_decay = 0
        self.factor = factor
        self.dataset = dataset
        if batch_size is None or batch_size <= 0:
            batch_size = task_to_batch_size[self.dataset]
        self.batch_size = batch_size
        self.device = device

        self.model, self.tokenizer = get_base_model(self.dataset)
        self.model.eval()
        self.model.to(self.device)
        self.model_bert = self.model.bert

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
        }[self.dataset]
        self.test_dataloader = get_dataloader(
            self.dataset, self.tokenizer, self.batch_size, split=split)
        self.epochs = task_to_epochs[self.dataset]
        if wiki_train:
            self.wiki_dataset = WikitextBatchLoader(batch_size=6, tokenizer=self.tokenizer)
            self.lr = 2e-5
            self.weight_decay = 5e-4
            self.epochs = wiki_epochs

        self.approx_bert = sparse.ApproxBertModel(self.model.config, factor=factor)
        self.approx_bert.train()
        self.approx_bert.to(self.device)
        self.optimizer = optim.Adam(self.approx_bert.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()

        self.last_metric_score = None
        self.last_loss = None

        print('Trainer: Save checkpoint path', self.checkpoint_path())

    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def eval_base_model(self, model = None, amp = False):
        self.seed()
        if model is None:
            model = self.model
        model.eval()
        
        metric = load_metric('glue', self.dataset)
        avg_length = 0
        
        for i, batch in enumerate(tqdm.tqdm(self.test_dataloader, desc='eval')):
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
            #if i > 100: break
        
        score = metric.compute()
        self.last_metric_score = score
        print('metric score', score)
        print('avg occupy', avg_length / len(self.test_dataloader))
        torch.cuda.empty_cache()
        return score

    def checkpoint_path(self):
        if self.wiki_train: 
            if self.wiki_epochs == 3:
                return f'saves/glue-{self.dataset}-{self.factor}-wiki.pth'
            return f'saves/glue-{self.dataset}-{self.factor}-wiki-b{self.wiki_epochs}.pth'
        return f'saves/glue-{self.dataset}-{self.factor}.pth'
    
    def load(self):
        state = torch.load(self.checkpoint_path(), map_location='cpu')
        self.model.load_state_dict(state['bert'])
        self.approx_bert.load_state_dict(state['approx_bert'])
        if 'last_metric_score' in state: self.last_metric_score = state['last_metric_score']
        if 'last_loss' in state: self.last_loss = state['last_loss']
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

    def eval_sparse_model(self, ks=0.5, use_forward=False):
        self.seed()
        wrapped_bert = sparse.ApproxSparseBertModel(self.model_bert, approx_bert=self.approx_bert, ks=ks)
        wrapped_bert.use_forward_sparse = use_forward
        sparse_cls_bert = berts.BertForSequenceClassification(self.model_bert.config)
        sparse_cls_bert.load_state_dict(self.model.state_dict())
        sparse_cls_bert.bert = wrapped_bert
        sparse_cls_bert.to(self.device).eval()
        
        sparse_result = self.eval_base_model(model = sparse_cls_bert)
        return sparse_result

    def eval_main(self, ks='dynamic'):
        self.load()
        
        bert_result = self.eval_base_model(model = self.model)
        sparse_result = self.eval_sparse_model(ks = ks)
        est_k = sparse.benchmark_get_average('est_k')
        print('est_k', est_k)

        print(bert_result, sparse_result)

    def main(self):
        #self.eval_base_model()

        for epoch in range(self.epochs):
            if self.wiki_train:
                pbar = tqdm.tqdm(self.wiki_dataset.__iter__())
            else:
                pbar = tqdm.tqdm(self.train_dataloader)
            torch.cuda.empty_cache()
            for i, batch in enumerate(pbar):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                batch['output_attentions'] = True
                if 'labels' in batch: del batch['labels']
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    attentions = self.model(**batch).attentions
                
                with torch.cuda.amp.autocast():
                    approx_attentions = self.approx_bert(**batch).attentions
                    loss = 0
                    for j in range(len(attentions)):
                        loss += torch.mean(torch.square(approx_attentions[j]- attentions[j]))
                    loss /= len(attentions)
                self.scaler.scale(loss).backward()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if i % 10 == 0: pbar.set_description(f"[{epoch+1}/{self.epochs}] {loss:.6f}")
            
            loss_sum = 0
            loss_count = 0
            self.approx_bert.eval()
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
                
                loss_sum += loss
                loss_count += 1
            self.approx_bert.train()
            
            self.last_loss = loss.item()
            self.save()

def main(args):
    trainer = GlueAttentionApproxTrainer(
        args.subset, 
        factor=args.factor, 
        batch_size=args.batch_size, 
        device=args.device, 
        wiki_train=args.wiki
    )

    if args.eval:
        trainer.eval_main()
    else:
        trainer.main()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--factor', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--not-wiki', action='store_true', default=False)

    args = parser.parse_args()
    args.wiki = not args.not_wiki
    #ngpus = torch.cuda.device_count()
    #args.device = args.device % ngpus
    print(args)
    
    main(args)
