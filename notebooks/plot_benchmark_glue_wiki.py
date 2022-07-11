# %%
import trainer.glue_base as glue_base
import models.sparse_token as sparse
from utils.glue import get_score
import pickle, importlib
importlib.reload(glue_base)
importlib.reload(sparse)
Glue = glue_base.GlueAttentionApproxTrainer
PICKLE_PATH = "glue_benchmark_wiki.pkl"
TEX_PATH = "saves_plot/glue_benchmark_wiki.tex"

# %%
subsets = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli",]
subsets = ["mnli","mrpc","qnli","qqp","rte","sst2","stsb"]
kss = [
    0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.999, 'dynamic', 
    #'dynamic:avg:avg:true', 'dynamic:avg:avg:false', 'dynamic:avg:max:true', 'dynamic:avg:max:false',
    #'dynamic:max:avg:true', 'dynamic:max:avg:false', 'dynamic:max:max:true', 'dynamic:max:max:false',
]
#kss = ['dynamic']
sparse.benchmark_reset()
# subsets = ["mrpc"]
# kss = ['dynamic:avg:avg:f',0.1]

results = {}
i = 0
for subset in subsets:
    trainer = Glue(dataset=subset, factor=16, batch_size=-1, device=0, wiki_train=True)
    trainer.load()
    scores = {}
    metric_name = ""
    bert_score, metric_name = get_score(trainer.eval_base_model())
    scores['bert'] = f'{bert_score:.5f}'
    for ks in kss:
        sparse.benchmark_reset()
        sparse_score, _ = get_score(trainer.eval_sparse_model(ks=ks))
        if isinstance(ks, str) and ks.startswith('dynamic'):
            est_k = sparse.benchmark_get_average('est_k')
            scores[str(ks)] = f'{sparse_score:.5f} (k:{est_k:.2f})'
        else:
            scores[str(ks)] = f'{sparse_score:.5f}'
        i += 1
        count = len(subsets) * len(kss)
        print(f'{i}/{count}')
    results[f"{subset} ({metric_name})"] = scores

with open(PICKLE_PATH, 'wb') as f:
    pickle.dump(results, f)

sparse.benchmark_report()

# %%
import pickle
import pandas as pd

with open(PICKLE_PATH, 'rb') as f:
    results = pickle.load(f)
with open('glue_benchmark.pkl', 'rb') as f:
    results_original = pickle.load(f)

def convert_data_to_results(results):
    data = []
    subsets = list(results.keys())
    factors = list(results[subsets[0]].keys())
    for factor in factors:             
        row = []
        for subset in subsets:
            row.append(results[subset][factor])
        data.append(row)
    return data, factors

data, factors = convert_data_to_results(results)
data_origin, _ = convert_data_to_results(results_original)
data_wo_data_augment = ("0.53388 (k:0.48)	0.81559 (k:0.30)	0.74667 (k:0.66)	0.90445 (k:0.41)	"+\
    "0.90893 (k:0.43)	0.70758 (k:0.38)	0.92202 (k:0.69)	0.86572 (k:0.49)	0.56338 (k:0.54)").split("\t")

factors[-1] = "dynamic:w_augment"
indicies = factors+["dynamic:original", "dynamic:wo_augment"]
columns = subsets[:]
df_data = data + [data_origin[9], data_wo_data_augment]

#calculate reproducibility
data_scalar = []
for line in df_data:
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
    df_data[i].append(f"{r*100:.2f}")
columns.append("reproduce")

df = pd.DataFrame(
    df_data,
    columns=columns, 
    index=indicies
)
#df = df.reindex(["bert", "dynamic:original", "dynamic:w_augment", "dynamic:wo_augment"])
tex = df.to_latex()
with open(TEX_PATH, 'w') as f:
    f.write(tex)
df

# %% [markdown]
# 	cola	mnli	mrpc	qnli	qqp	rte	sst2	stsb	wnli	reproduce
# bert	0.53388	0.84198	0.84406	0.91543	0.90908	0.72563	0.92431	0.88047	0.56338	100.00
# dynamic:original	0.53388 (k:0.48)	0.84004 (k:0.33)	0.83942 (k:0.71)	0.91287 (k:0.44)	0.90920 (k:0.45)	0.72563 (k:0.55)	0.92317 (k:0.73)	0.88043 (k:0.51)	0.56338 (k:0.63)	99.87
# dynamic:w_augment	0.53388 (k:0.48)	0.83647 (k:0.33)	0.83826 (k:0.72)	0.91506 (k:0.48)	0.90905 (k:0.45)	0.66426 (k:0.40)	0.92202 (k:0.68)	0.88034 (k:0.51)	0.56338 (k:0.61)	98.88
# dynamic:wo_augment	0.53388 (k:0.48)	0.81559 (k:0.30)	0.74667 (k:0.66)	0.90445 (k:0.41)	0.90893 (k:0.43)	0.70758 (k:0.38)	0.92202 (k:0.69)	0.86572 (k:0.49)	0.56338 (k:0.54)	97.74

# %% [markdown]
# # pre data augment
# 
# cola (matthews_correlation)	mnli (acc)	mrpc (acc)	qnli (acc)	qqp (acc)	rte (acc)	sst2 (acc)	stsb (pearson)	wnli (acc)
# 
# bert	0.53388	0.84198	0.84406	0.91543	0.90908	0.72563	0.92431	0.88047	0.56338
# 
# dynamic	0.53388 (k:0.48)	0.81559 (k:0.30)	0.74667 (k:0.66)	0.90445 (k:0.41)	0.90893 (k:0.43)	0.70758 (k:0.38)	0.92202 (k:0.69)	0.86572 (k:0.49)	0.56338 (k:0.54)
# 
# dynamic:original	0.53388 (k:0.48)	0.84004 (k:0.33)	0.83942 (k:0.71)	0.91287 (k:0.44)	0.90920 (k:0.45)	0.72563 (k:0.55)	0.92317 (k:0.73)	0.88043 (k:0.51)	0.56338 (k:0.63)

# %%



