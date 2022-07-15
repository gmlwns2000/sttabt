import argparse, json, math
from matplotlib import pyplot as plt
from utils.glue import get_score
import torch, pickle

import trainer.concrete_trainer as concrete

p_logits = [-2, -1.5, -1, -0.5, 0, 3, 6]
epoch_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.4]
# p_logits = [0, 3]
# epoch_factors = [0.5, 0.2]

#p_logits = [-1]
#epoch_factors = [1.0 for _ in range(100)]

def plot(occupies, metrics, metric_name, subset, plot_name):
    with open('saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle', 'rb') as f:
        data = pickle.load(f)
    
    ks = [item[1] for item in data.keys() if item[0] == subset and item[1] != 'bert']
    xs_forward = xs_absatt = [data[(subset, k)]['occupy'] for k in ks]
    ys_absatt = [data[(subset, k)]['score_sparse'] for k in ks]
    ys_forward = [data[(subset, k)]['score_forward'] for k in ks]
    xs_sparse = [data[(subset, k)]['occupy_approx'] for k in ks]
    ys_sparse = [data[(subset, k)]['score_sparse_approx'] for k in ks]
    ys_bert = [data[(subset, 'bert')]['score_bert'] for _ in range(2)]
    all_xs = xs_forward + xs_sparse + occupies
    xs_bert = [min(all_xs), max(all_xs)]
    
    plt.style.use("seaborn")
    plt.clf()
    
    plt.plot(xs_absatt, ys_absatt, marker='o', label='sparse (abs.att.)')
    plt.plot(xs_sparse, ys_sparse, marker='o', label='sparse (approx.)')
    plt.plot(xs_forward, ys_forward, marker='o', label='forward only')
    plt.plot(xs_bert, ys_bert, linestyle='--', label='bert-base')
    plt.plot(occupies, metrics, marker='o', label='sparse (concrete)')
    
    plt.xlabel('occupy')
    plt.ylabel(metric_name)
    plt.title(f'{subset} ({metric_name})')
    plt.legend()
    plt.savefig(f'{plot_name}.png', dpi=300)

    with open(f'{plot_name}.json', 'w') as f:
        json.dump({
            'occupies': occupies,
            'metrics': metrics,
            'metric_name': metric_name,
            'subset': subset,
            'p_logits': p_logits,
            'xs_forward': xs_forward,
            'ys_forward': ys_forward,
            'xs_sparse': xs_sparse,
            'ys_sparse': ys_sparse,
            'xs_bert': xs_bert,
            'ys_bert': ys_bert,
            'xs_absatt': xs_absatt,
            'ys_absatt': ys_absatt,
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='cola')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--header', type=str, default='')
    args = parser.parse_args()
    subset = args.subset
    plot_name = f'saves_plot/concrete-glue-{args.header}{subset}'

    occupies = []
    metrics = []
    metric_name = None

    for i, p_logit in enumerate(p_logits):
        trainer = concrete.ConcreteTrainer(
            dataset=subset,
            factor=4,
            batch_size=args.batch_size
        )
        trainer.enable_checkpointing = False
        #trainer.reset_train()
        trainer.epochs = int(math.ceil(concrete.task_to_epochs[subset] * epoch_factors[i]))
        trainer.sparse_bert.module.bert.set_concrete_init_p_logit(p_logit)
        trainer.main()

        concrete.sparse.benchmark_reset()
        trainer.sparse_bert.module.bert.set_concrete_hard_threshold(0.5)
        result = trainer.eval_sparse_model(show_message=False)
        trainer.sparse_bert.module.bert.set_concrete_hard_threshold(None)
        occupy = concrete.sparse.benchmark_get_average('concrete_occupy')
        metric, metric_name = get_score(result)

        print(f'[{i+1}/{len(p_logits)}]({subset}) occupy: {occupy} metric: {metric}')
        occupies.append(occupy)
        metrics.append(metric)
    
    plot(occupies, metrics, metric_name, subset, plot_name)

if __name__ == '__main__':
    main()