import argparse, json, math, itertools
import gc
import os
from matplotlib import pyplot as plt
from utils import gpu_pool
from utils.glue import get_score
import torch, pickle
import trainer.concrete_trainer as concrete
from utils.gpu_pool import GPUPool, print
import multiprocessing as mp

VERSION="1"
GLUE_SUBSETS = "cola mrpc wnli stsb mnli qnli qqp rte sst2".split()

p_logits =      [-2.0, -1.5, -1.0, -0.5, 0.0, 1.0]
epoch_factors = [ 1.0,  1.0,  1.0,  1.0, 1.0, 1.0]
#p_logits = [0, 3]
# epoch_factors = [0.5, 0.2]

#p_logits = [-1]
#epoch_factors = [1.0 for _ in range(100)]

special_epoch_factors = {
    'cola': [1.0,   1.0,    1.0,    1.0,    0.6,    0.6],
    'wnli': [0.5,   0.4,    0.3,    0.2,    0.1,    0.1],
    'mnli': [0.5,   0.5,    0.5,    0.5,    0.5,    0.5],
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
    occupies_valid, losses_valid, metrics_valid,
    metric_name, subset, plot_name, dump={}
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
    all_xs = xs_forward + xs_sparse + occupies + occupies_no_train + occupies_valid
    xs_bert = [min(all_xs), max(all_xs)]
    
    plt.style.use("seaborn")
    plt.clf()
    
    plt.plot(xs_absatt, ys_absatt, marker='o', label='sparse (abs.att.)')
    plt.plot(xs_sparse, ys_sparse, marker='o', label='sparse (approx.)')
    plt.plot(xs_forward, ys_forward, marker='o', label='forward only')
    plt.plot(xs_bert, ys_bert, linestyle='--', label='bert-base')
    plt.plot(occupies, metrics, marker='o', label='sparse (concrete)')
    plt.plot(occupies_no_train, metrics_no_train, marker='o', label='sparse (concrete, no train)')
    plt.plot(occupies_valid, metrics_valid, marker='X', label='sparse (concrete, valid)')
    
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
            'occupies_valid': occupies_valid,
            'losses_valid': losses_valid,
            'metrics_valid': metrics_valid,
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
            'dump':dump
        }, f, indent=2)

def exp_p_logit(
    device, tqdm_position, 
    i, subset, factor, batch_size, p_logit,
    lr_multiplier=1.0, epochs_multiplier=1.0, grad_acc_multiplier=1.0, 
):
    gc.collect()
    torch.cuda.empty_cache()
    
    current_epoch_factors = special_epoch_factors.get(subset, epoch_factors)
    lr = 1e-5 if concrete.task_to_epochs[subset] * current_epoch_factors[i] >= 2.0 else (1e-5 * current_epoch_factors[i])
    lr *= lr_multiplier
    trainer = concrete.ConcreteTrainer(
        device = device,
        dataset = subset,
        factor = factor,
        batch_size = batch_size,
        lr = lr,
    )
    trainer.tqdm_position = tqdm_position
    trainer.tqdm_postfix = f'_{p_logit}_{factor}'
    trainer.enable_checkpointing = False
    trainer.gradient_accumulate_steps *= grad_acc_multiplier
    #trainer.reset_train()
    trainer.epochs = math.ceil(concrete.task_to_epochs[subset] * current_epoch_factors[i] * epochs_multiplier)
    trainer.epochs = int(max(trainer.epochs, 2))
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
    def exam_valid():
        concrete.sparse.benchmark_reset()
        trainer.set_concrete_hard_threshold(0.5)
        result, loss = trainer.eval_sparse_model(show_message=False, split='valid', return_loss=True)
        trainer.set_concrete_hard_threshold(None)
        occupy = concrete.sparse.benchmark_get_average('concrete_occupy')
        metric, _ = get_score(result)
        return occupy, metric, loss
    
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
    occupy_valid, metric_valid, loss_valid = exam_valid()
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
        'occupy_valid': occupy_valid,
        'metric_valid': metric_valid,
        'loss_valid': loss_valid,
        'subset': subset,
        'factor': factor, 
        'batch_size': batch_size, 
        'p_logit': p_logit,
        'lr_multiplier':lr_multiplier, 
        'epochs_multiplier':epochs_multiplier, 
        'grad_acc_multiplier':grad_acc_multiplier,
    }

def query_best_hyperparameter(args):
    # if already best validated hyperparameter is cached, then return from disk.
    # hyper parameter (lr mul, epochs mul, grad acc mul)
    # search space [0.5, 1.0, 2.0]
    json_path = f'saves_hparam/concrete-glue-plot-hyperparam-{args}.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            hparam = json.load(f)
            return (hparam['lr_mul'], hparam['epochs_mul'], hparam['grad_acc_mul'])
    else:
        cases = list(itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)))
        #cases = list(itertools.product((1.0,), (1.0,), (0.5, 1.0)))
        args_list = []
        for case in cases:
            args_list.append(args + case)
        
        pool = GPUPool()
        _results = pool.run(exp_p_logit, args_list)
        results = []
        for result in _results:
            results.append((
                result['loss_valid'], 
                (result['lr_multiplier'], result['epochs_multiplier'], result['grad_acc_multiplier'])
            ))
        results = sorted(results, key=lambda x: x[0], reverse=True)
        max_result = results[0]
        max_hparam = max_result[1]

        with open(json_path, 'w') as f:
            json.dump({
                'lr_mul': max_hparam[0],
                'epochs_mul': max_hparam[1],
                'grad_acc_mul': max_hparam[2],
                'loss_valid': max_result[0],
                'search_space': cases,
                'args': args,
                'raw_results': results,
            }, f, indent=2)
        
        print(f'Main.query_best_hparam: Find the best hparam {max_result}')

        return max_hparam

def exp(subset, batch_size, factor, plot_name):
    occupies = []
    flopses = []
    metrics = []
    occupies_no_train = []
    flopses_no_train = []
    metrics_no_train = []
    occupies_valid = []
    losses_valid = []
    metrics_valid = []
    metric_name = None

    args_list = []
    for i, p_logit in enumerate(p_logits):
        args = (i, subset, factor, batch_size, p_logit)
        hparam = query_best_hyperparameter(args)
        print(f'Main.exp: HParam {hparam} of args {args}')
        args_list.append(args + hparam)
    
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
        occupies_valid.append(r['occupy_valid'])
        losses_valid.append(r['loss_valid'])
        metrics_valid.append(r['metric_valid'])
        metric_name = r['metric_name']

    plot(
        factor,
        occupies, flopses, metrics, 
        occupies_no_train, flopses_no_train, metrics_no_train,
        occupies_valid, losses_valid, metrics_valid,
        metric_name, subset, plot_name, dump={
            'args_list': args_list,
        }
    )

def main_all(args):
    import subprocess
    
    if args.subset == 'all':
        subsets = GLUE_SUBSETS
    else:
        subsets = args.subset.strip().split()
        if any([(not s in GLUE_SUBSETS) for s in subsets]):
            print(f'Main: Following given subsets are not GLUE subset {[s for s in subsets if not s in GLUE_SUBSETS]}')
            return
    
    def process_all(args, subsets):
        failed_subset = []
        for subset in subsets:
            retcode = subprocess.call((
                f"python -m main.concrete_glue_plot "+\
                f"--batch-size {args.batch_size} "+\
                f"--subset {subset} --factor {args.factor} "+\
                (f"--header \"{args.header}\"" if args.header != '' else '')
            ).split())
            if retcode != 0:
                print(f'Main: Process exited with error code {retcode}, retry {subset}')
                failed_subset.append(subset)
        return failed_subset
    
    retries = 5
    exp_subsets = subsets
    for retry in range(retries):
        failed_subsets = process_all(args, subsets)
        if len(failed_subsets) == 0:
            break
        else:
            if retry == (retries-1):
                print(f'Main: Retry failed. Following subsets could not be processed {failed_subsets}')
            else:
                print(f"Main: Retry following subsets {failed_subsets} (Retry: {retry+1})")
        subsets = failed_subsets
    
    if len(subsets) == 0:
        processed_subsets = set(exp_subsets) - set(subsets)
        print(f"Main: Following subsets are successfully processed! {processed_subsets}")

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
    elif not args.subset in GLUE_SUBSETS:
        if len(args.subset.split()) > 1:
            main_all(args)
        else:
            raise Exception('Given subset is not GLUE', args.subset)
    else:
        exp(
            subset=args.subset,
            batch_size=args.batch_size,
            factor=args.factor,
            plot_name=plot_name
        )

if __name__ == '__main__':
    mp.set_start_method('spawn')
    gpu_pool.initialize()
    main()