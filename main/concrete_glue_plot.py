import argparse, json, math
from matplotlib import pyplot as plt
from utils.glue import get_score
import torch, pickle

import trainer.concrete_trainer as concrete

VERSION="1"

p_logits =      [-2.0, -1.5, -1.0, -0.5, 0.0, 3.0]
epoch_factors = [ 1.0,  1.0,  1.0,  1.0, 1.0, 1.0]
#p_logits = [0, 3]
# epoch_factors = [0.5, 0.2]

#p_logits = [-1]
#epoch_factors = [1.0 for _ in range(100)]

def plot(
    occupies, flopses, metrics, 
    occupies_no_train, flopses_no_train, metrics_no_train,
    metric_name, subset, plot_name
):
    with open('saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle', 'rb') as f:
        data = pickle.load(f)
    
    ks = [item[1] for item in data.keys() if item[0] == subset and item[1] != 'bert']
    xs_forward = xs_absatt = [data[(subset, k)]['occupy'] for k in ks]
    ys_absatt = [data[(subset, k)]['score_sparse'] for k in ks]
    ys_forward = [data[(subset, k)]['score_forward'] for k in ks]
    xs_sparse = [data[(subset, k)]['occupy_approx'] for k in ks]
    ys_sparse = [data[(subset, k)]['score_sparse_approx'] for k in ks]
    ys_bert = [data[(subset, 'bert')]['score_bert'] for _ in range(2)]
    all_xs = xs_forward + xs_sparse + occupies + occupies_no_train
    xs_bert = [min(all_xs), max(all_xs)]
    
    plt.style.use("seaborn")
    plt.clf()
    
    plt.plot(xs_absatt, ys_absatt, marker='o', label='sparse (abs.att.)')
    plt.plot(xs_sparse, ys_sparse, marker='o', label='sparse (approx.)')
    plt.plot(xs_forward, ys_forward, marker='o', label='forward only')
    plt.plot(xs_bert, ys_bert, linestyle='--', label='bert-base')
    plt.plot(occupies, metrics, marker='o', label='sparse (concrete)')
    plt.plot(occupies_no_train, metrics_no_train, marker='o', label='sparse (concrete, no train)')
    
    plt.xlabel('occupy')
    plt.ylabel(metric_name)
    plt.title(f'{subset} ({metric_name})')
    plt.legend()
    plt.savefig(f'{plot_name}.png', dpi=300)

    with open(f'{plot_name}.json', 'w') as f:
        json.dump({
            'occupies': occupies,
            'flopses': flopses,
            'metrics': metrics,
            'occupies_no_train': occupies_no_train,
            'flopses_no_train': flopses_no_train,
            'metrics_no_train': metrics_no_train,
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

def exp(subset, batch_size, factor, plot_name):
    occupies = []
    flopses = []
    metrics = []
    occupies_no_train = []
    flopses_no_train = []
    metrics_no_train = []
    metric_name = None

    for i, p_logit in enumerate(p_logits):
        trainer = concrete.ConcreteTrainer(
            dataset = subset,
            factor = factor,
            batch_size = batch_size,
            lr = None if concrete.task_to_epochs[subset] * epoch_factors[i] >= 1.0 else (1e-5 * epoch_factors[i])
        )
        trainer.enable_checkpointing = False
        #trainer.reset_train()
        trainer.epochs = int(math.ceil(concrete.task_to_epochs[subset] * epoch_factors[i]))
        trainer.set_concrete_init_p_logit(p_logit)
        def exam():
            concrete.sparse.benchmark_reset()
            trainer.set_concrete_hard_threshold(0.5)
            result = trainer.eval_sparse_model(show_message=False)
            trainer.set_concrete_hard_threshold(None)
            occupy = concrete.sparse.benchmark_get_average('concrete_occupy')
            flops = concrete.sparse.benchmark_get_average('sparse_approx_flops')
            metric, metric_name = get_score(result)
            return occupy, flops, metric, metric_name
        #TODO: calc before train
        occupy_no_train, flops_no_train, metric_no_train, metric_name = exam()
        occupies_no_train.append(occupy_no_train)
        flopses_no_train.append(flops_no_train)
        metrics_no_train.append(metric_no_train)
        trainer.main()

        occupy, flops, metric, metric_name = exam()
        print(f'[{i+1}/{len(p_logits)}]({subset}) occupy: {occupy} flops: {flops} metric: {metric} occupy_no: {occupy_no_train} flops_no: {flops_no_train} metric_no: {metric_no_train}')
        occupies.append(occupy)
        flopses.append(flops)
        metrics.append(metric)
    
    plot(
        occupies, flopses, metrics, 
        occupies_no_train, flopses_no_train, metrics_no_train,
        metric_name, subset, plot_name
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='cola')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--header', type=str, default='')
    args = parser.parse_args()
    plot_name = f'saves_plot/concrete-glue-{args.header}{args.subset}'

    exp(
        subset=args.subset,
        batch_size=args.batch_size,
        factor=args.factor,
        plot_name=plot_name
    )

if __name__ == '__main__':
    main()