import transformers, torch, tqdm, random
import numpy as np
import transformers.models.bert.modeling_bert as berts
import models.sparse_token as sparse
from datasets import load_dataset, load_metric
from torch import optim, nn

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

def get_dataloader(subset, tokenizer, batch_size, split='train'):
    dataset = load_dataset('glue', subset, split=split)
    sentence1_key, sentence2_key = task_to_keys[subset]
    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=512, truncation=True)
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
        "mnli": "textattack/bert-base-uncased-MNLI",
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
    bert = model.from_pretrained(checkpoint)
    tokenizer = transformers.BertTokenizerFast.from_pretrained(checkpoint)
    return bert, tokenizer

class GlueAttentionApproxTrainer:
    def __init__(self, dataset, factor, batch_size=2, device=0):
        self.seed()

        self.factor = factor
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.bert, self.tokenizer = get_base_model(self.dataset)
        self.bert.eval()
        self.bert.to(self.device)
        self.train_dataloader = get_dataloader(self.dataset, self.tokenizer, self.batch_size)
        self.test_dataloader = get_dataloader(self.dataset, self.tokenizer, self.batch_size, split='test')
        
        self.epochs = 20
        self.approx_bert = sparse.ApproxBertModel(self.bert.config, factor=factor)
        self.approx_bert.train()
        self.approx_bert.to(self.device)
        self.optimizer = optim.Adam(self.approx_bert.parameters(), lr=5e-5)
        self.scaler = torch.cuda.amp.GradScaler()

        #self.eval_base_model()

    def seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def eval_base_model(self):
        metric = load_metric('glue', self.dataset)
        for i, batch in enumerate(tqdm.tqdm(self.test_dataloader)):
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.bert(**batch)
                predictions = outputs[1]
                metric.add_batch(predictions=torch.argmax(predictions, dim=-1), references=batch['labels'])
        score = metric.compute()
        print('metric score', score)

    def checkpoint_path(self):
        return f'saves/glue-{self.dataset}-{self.factor}.pth'
    
    def load(self):
        pass

    def save(self):
        torch.save({
            'bert':self.bert.state_dict(),
            'approx_bert': self.approx_bert.state_dict(),
            'epochs':self.epochs,
        }, self.checkpoint_path())
        print('saved')

    def main(self):
        for epoch in range(self.epochs):
            pbar = tqdm.tqdm(self.train_dataloader)
            for i, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch['output_attentions'] = True
                with torch.no_grad(), torch.cuda.amp.autocast():
                    attentions = self.bert(**batch).attentions
                del batch['labels']
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
                if i % 10 == 0: pbar.set_description(f"{loss:.6f}")
            self.save()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--factor', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    trainer = GlueAttentionApproxTrainer(
        args.subset, factor=args.factor, batch_size=args.batch_size, device=args.device)
    trainer.main()