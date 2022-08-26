import argparse, json, itertools
from matplotlib import pyplot as plt
from main.concrete_glue_plot import GLUE_SUBSETS
from utils.glue import get_score

import trainer.ltp_trainer as ltp
from utils.gpu_pool import GPUPool, print

# experiment options
lambdas = [0.001, 0.01, 0.1, 1.0, 5.0, 25.0]
temperatures = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
special_lambdas = {
    'mnli': [1e-5, 1e-2, 1.0, 15.0, ]
}
special_temperatures = {
    'mnli': [2e-4, 1e-3], #mnli took way too long time to train... (about 2 weeks)
}

# lambdas = [0.001, 0.01, ]
# temperatures = [1e-4, 2e-4 ]

def search_hparam_inner(device, tqdm_position, subset, batch_size, ltp_lambda, ltp_temperature):
    trainer = ltp.LtpTrainer(subset, batch_size=batch_size, device=device)
    trainer.reset_train()
    trainer.tqdm_position = tqdm_position
    trainer.enable_checkpointing = False
    trainer.sparse_bert.module.ltp_lambda = ltp_lambda
    trainer.sparse_bert.module.bert.set_ltp_temperature(ltp_temperature)
    trainer.main()

    ltp.sparse.benchmark_reset()
    trainer.sparse_bert.module.bert.set_ltp_prune_token_soft_pruning(False)
    result = trainer.eval_sparse_model(show_message=False, split='valid')
    occupy = ltp.sparse.benchmark_get_average('ltp_occupy')
    #metric, metric_name = get_score(result)
    metric = -result['loss']
    metric_name = 'minus_loss' # therefore, larger is better :)

    return {
        'subset': subset,
        'lambda': ltp_lambda,
        'temperature': ltp_temperature,
        'valid_metric': metric,
        'metric_name': metric_name,
        'valid_occupy': occupy,
    }

def search_hparam(subset, batch_size):
    cases = list(itertools.product(lambdas, temperatures))
    cases = list([(subset, batch_size,) + it for it in cases])
    pool = GPUPool(name='SearchHParam')
    raw = pool.run(search_hparam_inner, cases)

    # select max metric
    hparam = {}     # dict: lambda -> temperature
    results = {}    # dict: lambda -> temperature -> metric
    for result in raw:
        prev = hparam.get(result['lambda'], {'metric': -987654321, 'temperature': None})
        if prev['metric'] <= result['valid_metric']:
            prev['metric'] = result['valid_metric']
            prev['temperature'] = result['temperature']
        hparam[result['lambda']] = prev

        temps = results.get(result['lambda'], {})
        temps[result['temperature']] = {
            'metric': result['valid_metric'],
            'metric_name': result['metric_name'],
            'occupy': result['valid_occupy'],
        }
        results[result['lambda']] = temps
    
    pool.close()
    
    return hparam, results

def run_exp_inner(device, tqdm_position, subset, batch_size, ltp_lambda, ltp_temperature):
    trainer = ltp.LtpTrainer(subset, batch_size=batch_size, device=device)
    trainer.tqdm_position = tqdm_position
    trainer.reset_train()
    trainer.sparse_bert.module.ltp_lambda = ltp_lambda
    trainer.sparse_bert.module.bert.set_ltp_temperature(ltp_temperature)
    trainer.main()

    ltp.sparse.benchmark_reset()
    result = trainer.eval_sparse_model(show_message=False, split='test')
    test_metric, _ = get_score(result)
    test_occupy = ltp.sparse.benchmark_get_average('ltp_occupy')

    return {
        'subset': subset,
        'batch_size': batch_size,
        'lambda': ltp_lambda,
        'temperature': ltp_temperature,
        'test_metric': test_metric,
        'test_occupy': test_occupy,
    }

def run_exp(subset, batch_size, lambdas, hparam):
    cases = [(subset, batch_size, ld, hparam[ld]['temperature'])for ld in lambdas]
    pool = GPUPool(name='RunExp')
    results = pool.run(run_exp_inner, cases)
    results = sorted(results, key=lambda x: x['test_occupy'])
    metrics = []
    occupies = []
    for result in results:
        metrics.append(result['test_metric'])
        occupies.append(result['test_occupy'])
    return occupies, metrics

def main_subset(subset, batch_size):
    global temperatures, lambdas

    if subset in special_temperatures:
        temperatures = special_temperatures[subset]
    if subset in special_lambdas:
        lambdas = special_lambdas[subset]
    plot_name = f'saves_plot/ltp-glue-{subset}'

    hparam, hparam_results = search_hparam(subset, batch_size)
    print('Main: hparam results', hparam_results)
    print('Main: hparam', hparam)
    max_test_occupies, max_test_metrics = run_exp(subset, batch_size, lambdas=lambdas, hparam=hparam)

    plt.style.use("seaborn")
    plt.clf()
    plt.plot(max_test_occupies, max_test_metrics, label='LTP(test)')
    plt.xlabel('occupy')
    plt.title(subset)
    plt.legend()
    plt.savefig(plot_name+'.png', dpi=300)
    with open(plot_name+'.json', 'w') as f:
        json.dump({
            'max_test_occupies': max_test_occupies,
            'max_test_metrics': max_test_metrics,
            'lambdas': lambdas,
            'temperatures': temperatures,
            'hparam': hparam,
            'hparam_results': hparam_results,
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--batch-size', type=int, default=-1)
    args = parser.parse_args()
    subset = args.subset

    if subset in GLUE_SUBSETS:
        main_subset(subset, args.batch_size)
    else:
        if subset.strip() == 'all':
            subsets = GLUE_SUBSETS
        else:
            subsets = [s.strip() for s in subset.strip().split()]
            if any([not s in GLUE_SUBSETS for s in subsets]):
                print('Main: some subset is not GLUE', subsets)
                return
        
        print('Main: Multiple subsets will be processed', subsets)

        for subset in subsets:
            main_subset(subset, args.batch_size)

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()