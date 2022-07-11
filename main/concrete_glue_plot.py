import argparse, json
from matplotlib import pyplot as plt
from utils.glue import get_score

import trainer.concrete_trainer as concrete

p_logits = [-4, -2, -1, 0, 2, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='cola')
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--header', type=str, default='')
    args = parser.parse_args()
    subset = args.subset
    plot_name = f'saves_plot/concrete-glue-{args.header}{subset}'

    trainer = concrete.ConcreteTrainer(
        dataset=subset,
        factor=4,
        batch_size=args.batch_size
    )
    trainer.enable_checkpointing = False

    occupies = []
    metrics = []
    metric_name = None

    for i, p_logit in enumerate(p_logits):
        trainer.reset_train()
        trainer.sparse_bert.module.bert.set_concrete_init_p_logit(p_logit)
        trainer.main()

        concrete.sparse.benchmark_reset()
        result = trainer.eval_sparse_model(show_message=False)
        occupy = concrete.sparse.benchmark_get_average('concrete_occupy')
        metric, metric_name = get_score(result)

        print(f'[{i+1}/{len(p_logits)}]({subset}) occupy: {occupy} metric: {metric}')
        occupies.append(occupy)
        metrics.append(metric)
    
    plt.style.use("seaborn")

    plt.clf()
    plt.plot(occupies, metrics, label='test')
    plt.xlabel('occupy')
    plt.ylabel(metric_name)
    plt.title(subset)
    plt.legend()
    plt.savefig(f'{plot_name}.png', dpi=300)

    with open(f'{plot_name}.json', 'w') as f:
        json.dump({
            'occupies': occupies,
            'metrics': metrics,
            'metric_name': metric_name,
            'subset': subset,
            'p_logits': p_logits,
        }, f, indent=2)

if __name__ == '__main__':
    main()