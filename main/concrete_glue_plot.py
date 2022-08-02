import argparse, json, math
import gc
from matplotlib import pyplot as plt
from utils.glue import get_score
import torch, pickle
import trainer.concrete_trainer as concrete
from utils.gpu_pool import GPUPool
import multiprocessing as mp

VERSION="1"

p_logits =      [-2.0, -1.5, -1.0, -0.5, 0.0, 1.0]
epoch_factors = [ 1.0,  1.0,  1.0,  1.0, 1.0, 1.0]
#p_logits = [0, 3]
# epoch_factors = [0.5, 0.2]

#p_logits = [-1]
#epoch_factors = [1.0 for _ in range(100)]

special_epoch_factors = {
    'cola': [1.0,   1.0,    1.0,    1.0,    0.6,    0.6],
    'wnli': [0.5,   0.4,    0.3,    0.2,    0.1,    0.1],
    'mnli': [0.25,  0.25,   0.25,   0.25,   0.25,   0.25],
    'qnli': [0.25,  0.25,   0.25,   0.25,   0.25,   0.25],
}

factor_to_pickle = {
    4: 'saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle',
    8: 'saves_plot/[F8-PREWIKI.v2]glue_benchmark_accum_absatt.pickle',
}

def plot(
    factor,
    occupies, flopses, metrics, 
    occupies_no_train, flopses_no_train, metrics_no_train,
    metric_name, subset, plot_name
):
    with open(factor_to_pickle[factor], 'rb') as f:
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

def exp_p_logit(
    device, tqdm_position, 
    i, subset, factor, batch_size, p_logit
):
    gc.collect()
    torch.cuda.empty_cache()
    
    current_epoch_factors = special_epoch_factors.get(subset, epoch_factors)
    trainer = concrete.ConcreteTrainer(
        device = device,
        dataset = subset,
        factor = factor,
        batch_size = batch_size,
        lr = None if concrete.task_to_epochs[subset] * current_epoch_factors[i] >= 2.0 else (1e-5 * current_epoch_factors[i])
    )
    trainer.tqdm_position = tqdm_position
    trainer.tqdm_postfix = f'_{p_logit}_{factor}'
    trainer.enable_checkpointing = False
    #trainer.reset_train()
    trainer.epochs = int(max(math.ceil(concrete.task_to_epochs[subset] * current_epoch_factors[i]), 2))
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
    concrete.sparse.benchmark_sparse_approx_flops(True)
    concrete.sparse.benchmark_concrete_occupy(True)
    occupy_no_train, flops_no_train, metric_no_train, metric_name = exam()
    gc.collect()
    torch.cuda.empty_cache()

    concrete.sparse.benchmark_sparse_approx_flops(False)
    concrete.sparse.benchmark_concrete_occupy(False)
    trainer.main()
    gc.collect()
    torch.cuda.empty_cache()

    concrete.sparse.benchmark_sparse_approx_flops(True)
    concrete.sparse.benchmark_concrete_occupy(True)
    occupy, flops, metric, metric_name = exam()
    print(f'[{i+1}/{len(p_logits)}]({subset}) occupy: {occupy} flops: {flops} metric: {metric} occupy_no: {occupy_no_train} flops_no: {flops_no_train} metric_no: {metric_no_train}')

    return {
        'i': i,
        'metric_name': metric_name,
        'occupy_no_train':occupy_no_train,
        'flops_no_train': flops_no_train,
        'metric_no_train': metric_no_train,
        'occupy': occupy,
        'flops': flops,
        'metric': metric,
    }

def exp(subset, batch_size, factor, plot_name):
    occupies = []
    flopses = []
    metrics = []
    occupies_no_train = []
    flopses_no_train = []
    metrics_no_train = []
    metric_name = None

    args_list = []
    for i, p_logit in enumerate(p_logits):
        args_list.append((i, subset, factor, batch_size, p_logit))
    
    pool = GPUPool()
    results = pool.run(exp_p_logit, args_list)
    results = sorted(results, key=lambda it: it['i'])
    for r in results:
        occupies_no_train.append(r['occupy_no_train'])
        flopses_no_train.append(r['flops_no_train'])
        metrics_no_train.append(r['metric_no_train'])
        occupies.append(r['occupy'])
        flopses.append(r['flops'])
        metrics.append(r['metric'])
        metric_name = r['metric_name']

    plot(
        factor,
        occupies, flopses, metrics, 
        occupies_no_train, flopses_no_train, metrics_no_train,
        metric_name, subset, plot_name
    )

def main_all(args):
    import subprocess
    
    subsets = "cola mrpc wnli stsb mnli qnli qqp rte sst2".split()
    
    def process_all(args, subsets):
        failed_subset = []
        for subset in subsets:
            retcode = subprocess.call(
                f"python --batch-size {args.batch_size} "+\
                 "--subset {subset} --factor {args.factor} "+\
                 "--header \"{args.header}\"".split()
            )
            if retcode != 0:
                print(f'Main: Process exited with error code {retcode}, retry {subset}')
                failed_subset.append(subset)
        return failed_subset
    
    for retry in range(5):
        failed_subsets = process_all(args, subsets)
        if len(failed_subsets) == 0:
            break
        else:
            print(f"Main: Retry following subsets {failed_subsets}")
        subsets = failed_subsets
    
    if len(subsets) > 0:
        print(f"Main: Following subsets are failed to eval after retries. {subsets}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='cola')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--header', type=str, default='')
    args = parser.parse_args()
    plot_name = f'saves_plot/concrete-glue-{args.header}{args.subset}{"" if args.factor == 4 else f"-{args.factor}"}'

    if args.subset == 'all':
        main_all(args)
    else:
        exp(
            subset=args.subset,
            batch_size=args.batch_size,
            factor=args.factor,
            plot_name=plot_name
        )

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()