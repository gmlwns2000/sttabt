# %%
import trainer.glue_base as glue_base
import models.sparse_token as sparse
import pickle, importlib, itertools, gc
import torch
from matplotlib import pyplot as plt
importlib.reload(glue_base)
importlib.reload(sparse)
sparse.set_update_input_mask_accumulate_indices(True)

Glue = glue_base.GlueAttentionApproxTrainer

RESULT_PKL = 'glue_benchmark_accum_absatt.pkl'

# %%
subsets = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli",]
kss = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75]
# subsets = ['rte']
# kss = [0.05, 0.1,]

# %%
results = {}

def get_score(score):
    if 'accuracy' in score:
        return score['accuracy'], "acc"
    first_metric = list(score.keys())[0]
    return score[first_metric], first_metric

def run_exp():
    global subsets, kss

    cases = list(itertools.product(subsets, kss))
    current_subset = None
    trainer = None
    for i, (subset, ks) in enumerate(cases):
        if current_subset != subset:
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
            trainer = Glue(subset, 16, batch_size=8, wiki_train=False)
            bert_score, _ = get_score(trainer.eval_base_model())
            results[(subset, 'bert')] = { 'score_bert':bert_score }
            print('bert', bert_score)
            current_subset = subset
        
        trainer.set_batch_size(1)
        ksx = [ks for _ in range(12)]
        sparse.benchmark_reset()
        score_sparse, metric = get_score(trainer.eval_sparse_model(ks=ksx, run_original_attention=True, show_message=False))
        mask_occupy = sparse.benchmark_get_average('mask_occupy')
        print('sparse absatt', score_sparse)

        trainer.set_batch_size(8)
        target_ks = mask_occupy
        ksx = [target_ks*0.5+((1-x/11.0)**1.5) * target_ks for x in range(12)]
        score_forward, _ = get_score(trainer.eval_sparse_model(ks=ksx, use_forward=True, show_message=False))
        print('forward', score_forward)

        result = {
            'occupy':mask_occupy, 'score_sparse':score_sparse, 
            'score_forward':score_forward, 'metric':metric
        }
        print(f"RESULT {subset}@{ks} ({i+1}/{len(cases)}) |", result)
        results[(subset, ks)] = result

    with open('glue_benchmark_forward.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results
results = run_exp()

# %%
import pickle
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("seaborn")

with open('glue_benchmark_forward.pkl', 'rb') as f:
    results = pickle.load(f)

for subset in subsets:
    acc_sparse = []
    acc_forward = []
    occupy = []
    metric = None
    for ks in kss:
        item = results[(subset, ks)]
        metric = item['metric']
        acc_sparse.append(item['score_sparse'])
        acc_forward.append(item['score_forward'])
        occupy.append(item['occupy'])
    acc_bert = results[(subset, 'bert')]['score_bert']
    acc_bert = [acc_bert for _ in occupy]
    plt.plot(occupy, acc_sparse, marker='o', label='sparse (abs.att.)')
    plt.plot(occupy, acc_forward, marker='o', label='forward only')
    plt.plot(occupy, acc_bert, linestyle='--', label='bert-base')
    plt.xlabel('occupy')
    plt.ylabel(metric)
    plt.legend()
    plt.title(f'{subset} ({metric})')
    plt.savefig(f'saves_plot/accum_absolute_attention_{subset}.png', dpi=320)
    plt.show()

# %%



