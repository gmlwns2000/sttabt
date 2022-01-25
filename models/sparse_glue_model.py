import models.sparse_token as sparse
from trainer.glue_base import GlueAttentionApproxTrainer as GlueTrainer
import pickle

class SparseGlueWorker:
    def __init__(self, batch_size=4, device=1, subset='mrpc', factor=16) -> None:
        self.device = device
        self.batch_size = batch_size
        
        self.trainer = GlueTrainer(
            dataset=subset, 
            factor=factor, 
            batch_size=batch_size, 
            device=device
        )
        self.trainer.load()

        self.model = self.trainer.bert
        self.bert = self.trainer.bert.bert
        self.approx_bert = self.trainer.approx_bert

        self.model.eval().to(self.device)
        self.bert.eval().to(self.device)
        self.approx_bert.eval().to(self.device)
        self.wrapped_bert = sparse.ApproxSparseBertModel(
            self.bert, approx_bert=self.approx_bert, ks=0.5)
        self.wrapped_bert.eval().to(self.device)
    
    def run_eval(self, bert):
        self.model.bert = bert
        score = self.trainer.eval_base_model(model=self.model)
        return score

    def main(self):
        print('bert')
        bert_result = self.run_eval(self.bert)
        print('sparse bert')
        sparse_result = self.run_eval(self.wrapped_bert)
        
        reproduce = 0.0
        if 'accuracy' in bert_result:
            reproduce = sparse_result['accuracy'] / bert_result['accuracy']
        elif 'f1' in bert_result:
            reproduce = sparse_result['f1'] / bert_result['f1']
        else:
            raise Exception('unknown mark')

        with open('bench_result.pkl', 'wb') as f:
            pickle.dump({
                'bert':bert_result,
                'sparse':sparse_result,
                'reproduce':reproduce,
            }, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--factor', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    worker = SparseGlueWorker(
        batch_size=args.batch_size,
        device=args.device,
        subset=args.subset,
        factor=args.factor,
    )
    worker.main()