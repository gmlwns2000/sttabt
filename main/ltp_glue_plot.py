import argparse, random
from matplotlib import pyplot as plt

import trainer.ltp_trainer as ltp

# options
subset = 'mrpc'
lambdas = [0.01, 0.02, 0.035, 0.05, 0.1]
plot_name = f'saves_plot/ltp-glue-{subset}'

parser = argparse.ArgumentParser()
parser.add_argument('--subset', type=str, default=subset)
args = parser.parse_args()
subset = args.subset

def get_score(score):
    if 'accuracy' in score:
        return score['accuracy'], "acc"
    first_metric = list(score.keys())[0]
    return score[first_metric], first_metric

occupies = []
metrics = []
metric_name = None
for ld in lambdas:
    trainer = ltp.LtpTrainer(subset)
    trainer.sparse_bert.module.ltp_lambda = ld
    trainer.main()

    ltp.sparse.benchmark_reset()
    result = trainer.eval_sparse_model(show_message=False)
    occupy = ltp.sparse.benchmark_get_average('ltp_occupy')
    occupies.append(occupy)
    metric, metric_name = get_score(result)
    metrics.append(metric)

    print(f'({ld}) evaluate sparse net. score: {metric}, occupy: {occupy}')

plt.style.use("seaborn")
plt.clf()
points = [(occupies[i], metrics[i]) for i in range(len(occupies))]
points = sorted(points, key = lambda x: x[0])
plt.plot([p[0] for p in points], [p[1] for p in points], label='LTP')
plt.xlabel('occupy')
plt.ylabel(metric_name)
plt.legend()
plt.savefig(plot_name+'.png', dpi=300)