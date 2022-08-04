import argparse, json
from matplotlib import pyplot as plt
from utils.glue import get_score

import trainer.ltp_trainer as ltp

# options
lambdas = [0.001, 0.01, 0.1, 1.0, 5.0, 25.0]
temperatures = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
special_temperatures = {
    'mnli': [2e-4, 1e-3],
}
# lambdas = [0.001, 0.01, ]
# temperatures = [1e-4, 2e-4 ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--batch-size', type=int, default=-1)
    args = parser.parse_args()
    subset = args.subset
    if subset in special_temperatures:
        temperature = special_temperatures[subset]
    plot_name = f'saves_plot/ltp-glue-{subset}'

    occupies = []
    metrics = []
    max_test_occupies = []
    max_test_metrics = []
    metric_name = None
    trainer = ltp.LtpTrainer(subset, batch_size=args.batch_size)
    trainer.enable_checkpointing = False
    for il, ld in enumerate(lambdas):
        metrics_t = []
        occupies_t = []
        max_temperature = -1
        max_metric = -1
        for it, temperature in enumerate(temperatures):
            trainer.reset_train()
            trainer.sparse_bert.module.ltp_lambda = ld
            trainer.sparse_bert.module.bert.set_ltp_temperature(temperature)
            trainer.main()

            ltp.sparse.benchmark_reset()
            result = trainer.eval_sparse_model(show_message=False, split='valid')
            occupy = ltp.sparse.benchmark_get_average('ltp_occupy')
            occupies_t.append(occupy)
            metric, metric_name = get_score(result)
            metrics_t.append(metric)
            if metric >= max_metric:
                max_temperature = temperature
                max_metric = metric

            print(f'[{il*len(temperatures)+it+1}/{len(lambdas)*len(temperatures)}] (l:{ld}/t:{temperature}) evaluate sparse net. score: {metric}, occupy: {occupy}')
        
        #run test split
        trainer.reset_train()
        trainer.sparse_bert.module.ltp_lambda = ld
        trainer.sparse_bert.module.bert.set_ltp_temperature(max_temperature)
        trainer.main()
        
        ltp.sparse.benchmark_reset()
        result = trainer.eval_sparse_model(show_message=False, split='test')
        test_metric, _ = get_score(result)
        test_occupy = ltp.sparse.benchmark_get_average('ltp_occupy')

        max_test_occupies.append(test_occupy)
        max_test_metrics.append(test_metric)
        print(f'[(test){il+1}/{len(lambdas)}] (l:{ld}) eval best test. score: {test_metric}, occupy: {test_occupy} (valid. score: {metric}, occupy: {occupy})')

        occupies.append(occupies_t)
        metrics.append(metrics_t)

    plt.style.use("seaborn")
    plt.clf()
    def lavg(l): return sum(l) / len(l)
    def lmed(l): return sorted(l)[len(l)//2]
    points = [(
        lavg(occupies[i]),
        lmed(metrics[i]),
        lmed(metrics[i])-min(metrics[i]),
        max(metrics[i])-lmed(metrics[i])
    ) for i in range(len(occupies))]
    points = sorted(points, key = lambda x: x[0])
    # xs = [p[0] for p in points]
    # ys = [p[1] for p in points]
    # es = [
    #     list(p[2] for p in points),
    #     list(p[3] for p in points)
    # ]
    #plt.errorbar(xs, ys, yerr=es, label='LTP')
    xs = max_test_occupies
    ys = max_test_metrics
    plt.plot(xs, ys, label='LTP(test)')
    xs = []
    ys = []
    for i in range(len(occupies)):
        ots = occupies[i]
        mts = metrics[i]
        for j in range(len(ots)):
            xs.append(ots[j])
            ys.append(mts[j])
    plt.scatter(xs, ys, label='LTP(RAW)')
    plt.xlabel('occupy')
    plt.ylabel(metric_name)
    plt.title(subset)
    plt.legend()
    plt.savefig(plot_name+'.png', dpi=300)
    with open(plot_name+'.json', 'w') as f:
        json.dump({
            'occupies': occupies,
            'metrics': metrics,
            'max_test_occupies': max_test_occupies,
            'max_test_metrics': max_test_metrics,
            'lambdas': lambdas,
            'temperatures': temperatures
        }, f, indent=2)

if __name__ == '__main__':
    main()