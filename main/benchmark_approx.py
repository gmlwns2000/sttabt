import torch, random, math, time, sys, os, tqdm
import numpy as np
import numba
from trainer.classification import Trainer
from trainer.attention_approx import Trainer as ApproxTrainer
import importlib
import models.sparse_token as sparse

def run(
    model='bert-base', 
    target='sparse', 
    device='cuda', 
    batch_size=32,
    factor = 16
):
    trainer = Trainer(
        batch_size=batch_size, model='bert-base', device='cpu')
    trainer.load()
    trainer.model.eval()
    bert = trainer.model.bert
    fc = trainer.model.classifier
    batch = trainer.get_batch()
    test_batch = trainer.get_batch(test=False)

    def eval_fc(lm_output, fc=fc, batch=batch):
        last_hidden = lm_output.last_hidden_state[:,0,:]
        x = fc(last_hidden)
        return torch.argmax(x, dim=-1), batch.labels, lm_output

    def eval(bert, fc=fc, batch=batch):
        lm_output = bert(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_masks,
            output_hidden_states = True,
            output_attentions = True,
        )
        return eval_fc(lm_output, fc=fc, batch=batch)

    def approx_eval(sparse_bert, approx_bert, fc=fc, batch=batch):
        lm_output = sparse.run_bert_with_approx(
            sparse_bert, 
            approx_bert, 
            {
                'input_ids': batch.input_ids,
                'attention_mask': batch.attention_masks,
                'output_hidden_states': True,
                'output_attentions': True,
            },
            ks = [0.5]*12, #[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.35,0.35,0.15,],
        )
        return eval_fc(lm_output, fc=fc, batch=batch)

    def accuracy(batch_eval, N=7200//16, device=0):
        trainer.seed()
        trainer.dataset.batch_size = 16
        acc_sum = 0
        for i in tqdm.tqdm(range(N)):
            batch = trainer.get_batch(test=True).to(device)
            with torch.no_grad():
                output, label, _ = batch_eval(batch)
            acc_sum += torch.mean((output == label) * 1.0)
        return acc_sum.item() / N

    def benchmark(eval, batch_size=8, N=100, WARM=20, amp=False, device=0, end_warm=None):
        assert WARM < (N * 0.33)
        trainer.dataset.batch_size = batch_size
        batch = trainer.get_batch(test=True)
        batch = batch.to(device)
        assert batch.input_ids.shape[0] == batch_size
        for i in tqdm.tqdm(range(N)):
            if i == WARM: 
                t = time.time()
                if not end_warm is None: end_warm()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                eval(batch)
        t = time.time() - t
        t_item = t / (batch_size * (N-WARM))
        return t, t_item * 1000, 1.0/t_item
    
    batch = batch.to(device)
    test_batch = test_batch.to(device)
    
    benchmark_device = device
    benchmark_batch_size = batch_size

    if target == 'sparse':
        approx_trainer = ApproxTrainer(
            batch_size=batch_size, factor=factor, model=trainer.model_type, device='cpu')
        approx_trainer.load()
        approx_bert = approx_trainer.bert
        approx_bert = approx_bert.eval()
        
        sparse.timer_reset()

        sparse_bert = sparse.SparseBertModel(bert.config)
        sparse_bert.eval()
        sparse_bert.load_state_dict(bert.state_dict())
        sparse.set_print(sparse_bert, False)
        sparse.set_backup_last_inputs(sparse_bert, False)
        sparse.set_output_masking(sparse_bert, False)

        sparse_bert=sparse_bert.to(benchmark_device)
        approx_bert=approx_bert.to(benchmark_device)
        time_approx = benchmark(
            eval = lambda batch: approx_eval(sparse_bert, approx_bert, batch=batch, fc=lambda x: x),
            batch_size = benchmark_batch_size,
            WARM = 20,
            N = 1000,
            device = benchmark_device,
            end_warm = lambda: sparse.timer_reset()
        )
        sparse.timer_report()
        print(time_approx)
        print('acc', accuracy(
            lambda batch: approx_eval(sparse_bert, approx_bert, batch=batch), device=device)
        )
    elif target == 'bert':
        bert = bert.to(benchmark_device)
        time_bert = benchmark(
            lambda batch: eval(bert, batch=batch, fc=lambda x: x), 
            batch_size = benchmark_batch_size,
            WARM = 20,
            N = 1000,
            device = benchmark_device
        )
        print(time_bert)

if __name__ == '__main__':
    run(
        model = 'bert-base',
        target = 'sparse',
        device = 'cuda',
        batch_size = 32
    )