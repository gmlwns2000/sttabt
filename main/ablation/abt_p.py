import json
import models.sparse_token as sparse
from trainer import glue_base as glue
from trainer import concrete_trainer as concrete
from main.plot.constants import *

def get_metric(metric):
    metrics = ['acc', 'accuracy', 'matthews_correlation', 'f1_score']
    for m in metrics:
        if m in metric: return metric[m]
    raise Exception(metric)

def run_exp_with_p(pvalue, epochs=2, factor=4, subset='cola', p_logit=-0.5, batch_size=-1, method='abt', ks=0.1):
    sparse.set_abt_default_p(pvalue)
    if method == 'abt': 
        trainer = glue.GlueAttentionApproxTrainer(
            dataset=subset,
            factor=factor,
            batch_size=batch_size,
        )
        trainer.load()

        result = trainer.eval_sparse_model(ks = ks)

        return get_metric(result)
    else:
        trainer = concrete.ConcreteTrainer(
            dataset=subset,
            factor=factor,
            batch_size=batch_size
        )
        trainer.epochs = epochs
        trainer.enable_checkpointing = False
        trainer.set_concrete_init_p_logit(p_logit)

        concrete.sparse.benchmark_sparse_approx_flops(False)
        concrete.sparse.benchmark_concrete_occupy(False)
        trainer.main()
        concrete.sparse.benchmark_sparse_approx_flops(True)
        concrete.sparse.benchmark_concrete_occupy(True)

        result = trainer.eval_sparse_model(show_message=False)

        return get_metric(result)

JSON_PATH = './saves_plot/ablation-concrete-p.json'

def exp_all():
    results = {}

    subset = 'cola'
    for method in ['abt', 'concrete']:
        for factor in [4, 8]:
            metrics = []
            ps = [0.0, 0.1, 0.2, 0.5, 1.0]
            # ps = [0.1]
            for p in ps:
                m = run_exp_with_p(
                    p, subset=subset,
                    method=method,
                    p_logit=-0.5,
                    ks=0.1,
                    epochs=6,
                    factor=factor
                )
                print(f'AblationP: f(p:{p})={m}')
                metrics.append(m)
            print(list(zip(ps, metrics)))
            results[f'{method}.{factor}'] = {
                'subset': subset,
                'factor': factor,
                'method': method,
                'ps': ps,
                'metrics': metrics,
            }

    with open(JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)

def plot_all():
    import pandas as pd
    from matplotlib import pyplot as plt
    plt.style.use('seaborn-bright')

    with open(JSON_PATH, 'r') as f:
        results = json.load(f)

    df = pd.DataFrame()
    for method in ['abt', 'concrete']:
        for factor in [4, 8]:
            method_name = {
                'abt': 'ABT',
                'concrete': 'Concrete Masking',
            }[method]
            name = f'{method_name}@f{factor}'
            df[name] = results[f'{method}.{factor}']['metrics']
    df.index = results['abt.4']['ps']
    
    df.plot()
    plt.grid()
    plt.xlabel('$p$')
    plt.ylabel('Metric')
    plt.title(SUBSET_TO_NAME['cola'])

    plt.savefig('./saves_plot/ablation-concrete-p.pdf')
    plt.clf()

if __name__ == '__main__':
    # exp_all()
    plot_all()