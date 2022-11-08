import trainer.concrete_trainer as concrete
import torch, json

"""
#for NLP
concrete_loss_encoder_mask_avg_factor=1e-3
concrete_loss_factor=1e-3
#for ViT
concrete_loss_encoder_mask_avg_factor=100.0
concrete_loss_factor=1e-3

concrete_model.bert.encoder.concrete_loss_encoder_mask_avg_factor = 100.0
for layer in concrete_model.bert.encoder.layer:
    assert hasattr(layer, 'concrete_loss_factor')
    layer.concrete_loss_factor = 1e-3 # ease the factor, and let ratio decide it.
"""

def get_metric(metric):
    metrics = ['acc', 'accuracy', 'matthews_correlation', 'f1_score']
    for m in metrics:
        if m in metric: return metric[m]
    raise Exception(metric)

def exp_single(
    lambda_mask, lambda_p,
    subset, batch_size, epochs, p_logit, factor
):
    trainer = concrete.ConcreteTrainer(
        dataset=subset,
        factor=factor,
        epochs=epochs,
        batch_size=batch_size,
    )
    trainer.enable_checkpointing = False
    trainer.set_concrete_init_p_logit(p_logit)
    trainer.sparse_bert.module.bert.encoder.concrete_loss_encoder_mask_avg_factor = lambda_mask
    for layer in trainer.sparse_bert.module.bert.encoder.layer:
        assert hasattr(layer, 'concrete_loss_factor')
        layer.concrete_loss_factor = lambda_p
    
    concrete.sparse.benchmark_reset()
    concrete.sparse.benchmark_concrete_occupy(False)
    concrete.sparse.benchmark_sparse_approx_flops(False)
    trainer.main()
    
    concrete.sparse.benchmark_concrete_occupy(True)
    concrete.sparse.benchmark_sparse_approx_flops(True)
    result = trainer.eval_sparse_model()
    result['occupy'] = concrete.sparse.benchmark_get_average('concrete_occupy')
    return result

JSON_PATH = './saves_plot/ablation-concrete-lambda.json'

def exp_all(
    lambdas_mask = [1e-5, 1e-3, 1e-1], lambdas_p = [1e-5, 1e-3, 1e-1],
    subset='cola', batch_size=-1, epochs=6, p_logit=-0.5, factor=4,
):
    results = {}

    for mask in lambdas_mask:
        metrics = []
        occupies = []
        for p in lambdas_p:
            result = exp_single(mask, p, subset, batch_size, epochs, p_logit, factor)
            metric = get_metric(result)
            occupy = result['occupy']
            metrics.append(metric)
            occupies.append(occupy)
        results[f'mask{mask}'] = {
            'mask': mask,
            'lambda_ps': lambdas_p,
            'metrics': metrics,
            'occupies': occupies,
        }
    
    with open(JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)

import numpy as np
from matplotlib import pyplot as plt
def plot_all(
    lambdas_mask = [1e-5, 1e-3, 1e-1], lambdas_p = [1e-5, 1e-3, 1e-1],
):
    with open(JSON_PATH, 'r') as f:
        results = json.load(f)
    matrix = np.zeros((3, 3))
    for i in range(3):
        matrix[i] = np.array(results[f'mask{lambdas_mask[i]}']['metrics']) /\
            np.array(results[f'mask{lambdas_mask[i]}']['occupies'])
    print(matrix)

    fig, [ax, ax1] = plt.subplots(1, 2, tight_layout=True)

    ax.imshow(matrix)
    # fig.colorbar()

    ax.set_yticks(range(3))
    ax.set_yticklabels(lambdas_mask)
    ax.set_ylabel('$\lambda_{mask}$')

    ax.set_xticks(range(3))
    ax.set_xticklabels(lambdas_p)
    ax.set_xlabel('$\lambda_{p}$')

    ax.set_title('CoLA (Metric/Token Retention Ratio)')

    matrix = np.zeros((3, 3))
    for i in range(3):
        matrix[i] = np.array(results[f'mask{lambdas_mask[i]}']['metrics'])
    print(matrix)
    ax1.imshow(matrix)

    ax1.set_yticks(range(3))
    ax1.set_yticklabels(lambdas_mask)
    ax1.set_ylabel('$\lambda_{mask}$')

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(lambdas_p)
    ax1.set_xlabel('$\lambda_{p}$')

    ax1.set_title('CoLA (Metric)')

    fig.tight_layout()

    plt.savefig('./saves_plot/ablation-concrete-lambda.pdf', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    # exp_all()s
    plot_all()