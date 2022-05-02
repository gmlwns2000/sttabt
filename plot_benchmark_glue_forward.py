# %%
import trainer.glue_base as glue_base
import models.sparse_token as sparse
import pickle, importlib
importlib.reload(glue_base)
importlib.reload(sparse)
Glue = glue_base.GlueAttentionApproxTrainer

# %%
subsets = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli",]
subsets = ["mnli","mrpc","qnli","qqp","rte","sst2","stsb"]
kss = [
    0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.999,
]
sparse.benchmark_reset()
# subsets = ["mrpc"]
# kss = ['dynamic:avg:avg:f',0.1]

def get_score(score):
    if 'accuracy' in score:
        return score['accuracy'], "acc"
    first_metric = list(score.keys())[0]
    return score[first_metric], first_metric

results = {}
i = 0
for subset in subsets:
    trainer = Glue(dataset=subset, factor=16, batch_size=-1, device=0)
    trainer.load()
    scores = {}
    metric_name = ""
    bert_score, metric_name = get_score(trainer.eval_base_model())
    scores['bert'] = f'{bert_score:.5f}'
    for ks in kss:
        sparse.benchmark_reset()
        sparse_score, _ = get_score(trainer.eval_sparse_model(ks=ks, use_forward=True))
        if isinstance(ks, str) and ks.startswith('dynamic'):
            est_k = sparse.benchmark_get_average('est_k')
            scores[str(ks)] = f'{sparse_score:.5f} (k:{est_k:.2f})'
        else:
            scores[str(ks)] = f'{sparse_score:.5f}'
        i += 1
        count = len(subsets) * len(kss)
        print(f'{i}/{count} | {subset} {ks} = {sparse_score:.5f}')
    results[f"{subset} ({metric_name})"] = scores

with open('glue_benchmark_forward.pkl', 'wb') as f:
    pickle.dump(results, f)

sparse.benchmark_report()

# %%
import pickle
import pandas as pd

with open('glue_benchmark_forward.pkl', 'rb') as f:
    results = pickle.load(f)

data = []
subsets = list(results.keys())
factors = list(results[subsets[0]].keys())
for factor in factors:             
    row = []
    for subset in subsets:
        row.append(results[subset][factor])
    data.append(row)

#calculate reproducibility
data_scalar = []
for line in data:
    xs = []
    for item in line:
        xs.append(float(item.split()[0]))
    data_scalar.append(xs)
reproducibilities = []
for i in range(len(data_scalar)):
    rsum = 0
    for k in range(len(data_scalar[i])):
        rsum += data_scalar[i][k]/data_scalar[0][k]
    rsum /= len(data_scalar[i])
    reproducibilities.append(rsum)
for i, r in enumerate(reproducibilities):
    data[i].append(f"{r*100:.2f}")
subsets.append("reproduce")

df = pd.DataFrame(data, columns=subsets, index=factors)
tex = df.to_latex()
with open('saves_plot/glue_benchmark_forward.tex', 'w') as f:
    f.write(tex)
df

# %%
trainer = Glue(dataset='qnli', factor=16, batch_size=-1, device=0)
trainer.load() 

# %%
trainer.model.bert = trainer.model_bert
trainer.eval_base_model()

# %%
# trainer.seed()
# import models.sparse_token as sparse
# import transformers.models.bert.modeling_bert as berts
# import importlib
# importlib.reload(sparse)

# wrapped_bert = sparse.ApproxSparseBertModel(trainer.model_bert, approx_bert=trainer.approx_bert, ks=0.1)
# sparse_cls_bert = berts.BertForSequenceClassification(trainer.model_bert.config)
# sparse_cls_bert.load_state_dict(trainer.model.state_dict())
# sparse_cls_bert.bert = wrapped_bert
# sparse_cls_bert.to(trainer.device).eval()

# trainer.eval_base_model(model = sparse_cls_bert)

# %%


