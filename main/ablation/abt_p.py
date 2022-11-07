import models.sparse_token as sparse
from trainer import concrete_trainer as concrete

def get_metric(metric):
    metrics = ['acc', 'accuracy', 'matthew_correlation', 'f1_score']
    for m in metrics:
        if m in metric: return metric[m]
    raise Exception(metric)

def run_exp_with_p(pvalue, epochs=3, factor=4, subset='mrpc'):
    # sparse.set_abt_default_p(pvalue)
    trainer = concrete.ConcreteTrainer(
        dataset=subset,
        factor=factor,
    )
    trainer.epochs = epochs
    trainer.enable_checkpointing = False

    concrete.sparse.benchmark_sparse_approx_flops(False)
    concrete.sparse.benchmark_concrete_occupy(False)
    trainer.main()
    concrete.sparse.benchmark_sparse_approx_flops(True)
    concrete.sparse.benchmark_concrete_occupy(True)

    result = trainer.eval_sparse_model(show_message=False)

    return get_metric(result)

metrics = []
ps = [0.1, 0.1, 0.2, 0.5, 1.0]
for p in ps:
    m = run_exp_with_p(p)
    print(f'AblationP: f(p:{p})={m}')
    metrics.append(m)

print(zip(ps, metrics))