import argparse, json
from matplotlib import pyplot as plt

import trainer.ltp_trainer as ltp

# options
lambdas = [0.001, 0.01, 0.1, 1.0, 5.0]
temperatures = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='mrpc')
    parser.add_argument('--batch-size', type=int, default=-1)
    args = parser.parse_args()
    subset = args.subset
    plot_name = f'saves_plot/ltp-glue-{subset}'

    def get_score(score):
        if 'accuracy' in score:
            return score['accuracy'], "acc"
        first_metric = list(score.keys())[0]
        return score[first_metric], first_metric

    occupies = []
    metrics = []
    metric_name = None
    for il, ld in enumerate(lambdas):
        metrics_t = []
        for it, temperature in enumerate(temperatures):
            trainer = ltp.LtpTrainer(subset, batch_size=args.batch_size)
            trainer.sparse_bert.module.ltp_lambda = ld
            trainer.sparse_bert.module.set_ltp_temperature = temperature
            trainer.main()

            ltp.sparse.benchmark_reset()
            result = trainer.eval_sparse_model(show_message=False)
            occupy = ltp.sparse.benchmark_get_average('ltp_occupy')
            occupies.append(occupy)
            metric, metric_name = get_score(result)
            metrics_t.append(metric)

            print(f'[{il*len(temperatures)+it+1}/{len(lambdas)*len(temperatures)}] ({ld}/{temperature}) evaluate sparse net. score: {metric}, occupy: {occupy}')
        metrics.append(metrics_t)

    plt.style.use("seaborn")
    plt.clf()
    def lcenter(l): return l[len(l)//2]
    points = [(
        occupies[i],
        lcenter(metrics[i]),
        lcenter(metrics[i])-min(metrics[i]),
        max(metrics[i])-lcenter(metrics[i])
    ) for i in range(len(occupies))]
    points = sorted(points, key = lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    es = [
        (p[2] for p in points),
        (p[3] for p in points)
    ]
    plt.errorbar(xs, ys, yerr=es, label='LTP', uplims=True, lolims=True)
    plt.xlabel('occupy')
    plt.ylabel(metric_name)
    plt.title(subset)
    plt.legend()
    plt.savefig(plot_name+'.png', dpi=300)
    with open(plot_name+'.json', 'w') as f:
        json.dump({
            'occupies': occupies,
            'metrics': metrics,
            'lambdas': lambdas,
            'temperatures': temperatures
        }, f)

if __name__ == '__main__':
    main()