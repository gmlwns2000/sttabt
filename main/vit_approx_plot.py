import json
from matplotlib import pyplot as plt
plt.style.use("seaborn")
from trainer.vit_approx_trainer import VitApproxTrainer

model = 'deit-small'
subsets = ['base']
kxs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
EXPORT_NAME = './saves_plot/vit-approx'

def exam_subset(subset, factor, kxs):
    trainer = VitApproxTrainer(
        subset=subset, factor=factor, model=model,
    )
    trainer.load()

    batch_size = trainer.batch_size
    metric_name = 'accuracy'
    metric_baseline = trainer.eval_model_metric_base()
    
    occupies_sparse = []
    metrics_sparse = []
    occupies_forward = []
    metrics_forward = []
    occupies_absatt = []
    metrics_absatt = []

    for ik, kx in enumerate(kxs):
        trainer.set_batch_size(1)
        metric_sparse = trainer.eval_model_metric_sparse(ks=kx, mode='sparse')
        metrics_sparse.append(metric_sparse[metric_name])
        occupies_sparse.append(metric_sparse['occupy'])

        target_ks = metric_sparse['occupy']
        if target_ks <= 0.666:
            exam_ks = [target_ks*0.5+((1-x/10.0)**1.0) * target_ks for x in range(12)]
        else:
            exam_ks = [(1-x/10.0)*(2-2*target_ks)+(2*target_ks-1) for x in range(12)]
        trainer.set_batch_size(batch_size)
        metric_forward = trainer.eval_model_metric_sparse(ks=exam_ks, mode='forward')
        metrics_forward.append(metric_forward[metric_name])
        occupies_forward.append(metric_forward['occupy'])

        trainer.set_batch_size(1)
        metric_absatt = trainer.eval_model_metric_sparse(ks=kx, mode='absatt')
        metrics_absatt.append(metric_absatt[metric_name])
        occupies_absatt.append(metric_absatt['occupy'])

        print(
            f"[({subset}) {ik+1}/{len(kxs)}] "+\
            f"occ_forward: {metric_forward['occupy']} acc_forward: {metric_forward[metric_name]} | "+\
            f"occ_sparse: {metric_sparse['occupy']} acc_sparse: {metric_sparse[metric_name]} | "+\
            f"occ_absatt: {metric_absatt['occupy']} occ_absatt: {metric_absatt[metric_name]}"
        )
    
    return {
        'subset': trainer.subset,
        'factor': trainer.factor,
        'kxs': kxs,
        'metric_name': metric_name,
        'model': trainer.model_id,

        'metric_baseline': metric_baseline[metric_name],

        'metrics_sparse': metrics_sparse,
        'occupies_sparse': occupies_sparse,

        'metrics_forward': metrics_forward,
        'occupies_forward': occupies_forward,

        'metrics_absatt': metrics_absatt,
        'occupies_absatt': occupies_absatt,
    }

def exam_all(subsets, factor, kxs):
    data = {}
    for subset in subsets:
        ret = exam_subset(subset=subset, factor=factor, kxs=kxs)
        data[subset] = ret
    return data

def plot_exam(subset_name, data):
    plot_name = EXPORT_NAME + f'-{subset_name}'
    json_path = plot_name + '.json'
    png_path = plot_name + '.png'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    plt.clf()

    plt.plot(data['occupies_absatt'], data['metrics_absatt'], marker='o', label='sparse (abs.att.)')
    plt.plot(data['occupies_sparse'], data['metrics_sparse'], marker='o', label='sparse (approx.)')
    plt.plot(data['occupies_forward'], data['metrics_forward'],marker='o', label='forward only')
    occupies = data['occupies_absatt'] + data['occupies_sparse'] + data['occupies_forward']
    plt.plot([min(occupies), max(occupies)], [data['metric_baseline'],]*2, linestyle='--', label='vit-base')

    plt.legend()
    plt.xlabel('occupy')
    metric_name = data['metric_name']
    plt.ylabel(metric_name)
    plt.title(f'{subset_name} ({metric_name})')
    plt.savefig(png_path, dpi=320)

def plot_all(data):
    subsets = list(data.keys())
    for subset in subsets:
        plot_exam(subset, data[subset])

def main():
    data = exam_all(subsets=subsets, factor=4, kxs=kxs)
    plot_all(data=data)

if __name__ == '__main__':
    main()