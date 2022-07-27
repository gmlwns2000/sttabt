import trainer.glue_base as glue_base
import models.sparse_token as sparse
from utils.glue import get_score
import pickle, importlib, itertools, gc, json
import torch
from matplotlib import pyplot as plt
plt.style.use("seaborn")
sparse.set_update_input_mask_accumulate_indices(True)

Glue = glue_base.GlueAttentionApproxTrainer

plot_header = '[F4-PREWIKI.v2]'
result_name = None
result_pkl = None
def update_result_names():
    global plot_header, result_name, result_pkl
    result_name = 'saves_plot/' + plot_header + 'glue_benchmark_accum_absatt'
    result_pkl = result_name + '.pickle'
update_result_names()

factor = 4
subsets = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli",]
kss = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75]

def run_exp(run_approx=True):
    global subsets, kss

    results = {}
    cases = list(itertools.product(subsets, kss))
    current_subset = None
    trainer = None
    for i, (subset, ks) in enumerate(cases):
        if current_subset != subset:
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
            trainer = Glue(subset, factor=factor, batch_size=1, wiki_train=False)
            if run_approx: trainer.load()
            bert_score, _ = get_score(trainer.eval_base_model())
            results[(subset, 'bert')] = { 'score_bert':bert_score }
            print('bert', bert_score)
            current_subset = subset
        
        trainer.set_batch_size(1)
        ksx = [ks for _ in range(12)]
        sparse.benchmark_reset()
        score_sparse, metric = get_score(trainer.eval_sparse_model(ks=ksx, run_original_attention=True, show_message=False))
        mask_occupy = sparse.benchmark_get_average('mask_occupy')
        print('sparse absatt', score_sparse, '@', mask_occupy)

        if run_approx:
            trainer.set_batch_size(1)
            ksx = [ks for _ in range(12)]
            sparse.benchmark_reset()
            score_approx, metric = get_score(trainer.eval_sparse_model(ks=ksx, run_original_attention=False, show_message=False))
            mask_occupy_approx = sparse.benchmark_get_average('mask_occupy')
            print('sparse approx', score_approx, '@', mask_occupy_approx)

        trainer.set_batch_size(8)
        target_ks = mask_occupy
        if target_ks <= 0.666:
            ksx = [target_ks*0.5+((1-x/10.0)**1.0) * target_ks for x in range(12)]
        else:
            ksx = [(1-x/10.0)*(2-2*target_ks)+(2*target_ks-1) for x in range(12)]
        #ksx[-1] = 0.99
        sparse.benchmark_reset()
        score_forward, _ = get_score(trainer.eval_sparse_model(ks=ksx, use_forward=True, show_message=False))
        mask_occupy_forward = sparse.benchmark_get_average('forward_occupy')
        print('forward', score_forward, '@', mask_occupy_forward)

        result = {
            'occupy':mask_occupy, 'score_sparse':score_sparse, 
            'occupy_forward': mask_occupy_forward, 'score_forward':score_forward, 'metric':metric
        }
        if run_approx:
            result['score_sparse_approx'] = score_approx
            result['occupy_approx'] = mask_occupy_approx
        print(f"RESULT {subset}@{ks} ({i+1}/{len(cases)}) |", result)
        results[(subset, ks)] = result

    with open(result_pkl, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_from_last_pickle():
    with open(result_pkl, 'rb') as f:
        results = pickle.load(f)

    run_approx = 'occupy_approx' in results[results.keys()[0]]

    for subset in subsets:
        acc_sparse = []
        acc_approx = []
        acc_forward = []
        occupy = []
        occupy_approx = []
        occupy_forward = []
        metric = None
        for ks in kss:
            item = results[(subset, ks)]
            metric = item['metric']
            acc_sparse.append(item['score_sparse'])
            if run_approx: acc_approx.append(item['score_sparse_approx'])
            acc_forward.append(item['score_forward'])
            occupy.append(item['occupy'])
            occupy_forward.append(item['occupy_forward'])
            if run_approx: occupy_approx.append(item['occupy_approx'])
        acc_bert = results[(subset, 'bert')]['score_bert']
        occupy_bert = [min(occupy+occupy_approx), max(occupy+occupy_approx)]
        acc_bert = [acc_bert, acc_bert]
        plt.plot(occupy, acc_sparse, marker='o', label='sparse (abs.att.)')
        if run_approx: plt.plot(occupy_approx, acc_approx, marker='o', label='sparse (approx.)')
        plt.plot(occupy_forward, acc_forward, marker='o', label='forward only')
        plt.plot(occupy_bert, acc_bert, linestyle='--', label='bert-base')
        plt.xlabel('occupy')
        plt.ylabel(metric)
        plt.legend()
        plt.title(f'{subset} ({metric})')
        plt.savefig(f'{result_name}_{subset}.png', dpi=320)
        plt.show(block=False)
        plt.clf()

def main():
    global plot_header, factor

    #arg
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=str, default=None)
    parser.add_argument('--factor', type=int, default=None)

    args = parser.parse_args()
    if args.header is not None:
        plot_header = args.header
        update_result_names()
    if args.factor is not None:
        factor = args.factor
    
    run_exp()
    plot_from_last_pickle()

if __name__ == '__main__':
    main()