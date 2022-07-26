import json, math, random, pickle, os
from matplotlib import pyplot as plt
plt.style.use('seaborn-bright')

metric_to_name = {
    'acc': 'Accuracy (%)',
    'matthews_correlation': 'Matthews Correlation',
    'pearson': 'Pearson Correlation',
}

metric_to_scaler = {
    'acc': 100,
    'matthews_correlation': 100,
    'pearson': 1,
}

subset_to_name = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp":  "QQP",
    "rte":  "RTE",
    "sst2": "SST-2",
    "stsb": "STSB",
    "wnli": "WNLI",
}

def scale(lst, s):
    return list([it * s for it in lst])

def main():
    baseline_data_path = 'saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle'
    subsets = "cola mnli mrpc qnli qqp rte sst2 stsb wnli".split()
    skipped = []

    for subset in subsets:
        plot_name = f'saves_plot/combined-glue-{subset}'
        ltp_data_path = f'saves_plot/ltp-glue-{subset}.json'
        concrete_data_path = f'saves_plot/concrete-glue-{subset}.json'

        # load baselines
        with open(baseline_data_path, 'rb') as f:
            data = pickle.load(f)
        
        ks = [item[1] for item in data.keys() if item[0] == subset and item[1] != 'bert']
        xs_absatt = [data[(subset, k)]['occupy'] for k in ks]
        ys_absatt = [data[(subset, k)]['score_sparse'] for k in ks]
        xs_forward = [data[(subset, k)]['occupy_forward'] for k in ks]
        ys_forward = [data[(subset, k)]['score_forward'] for k in ks]
        xs_sparse = [data[(subset, k)]['occupy_approx'] for k in ks]
        ys_sparse = [data[(subset, k)]['score_sparse_approx'] for k in ks]
        ys_bert = [data[(subset, 'bert')]['score_bert'] for _ in range(2)]

        # load concrete
        if not os.path.exists(concrete_data_path):
            skipped.append((subset, 'please run concrete'))
            continue
        
        with open(concrete_data_path, 'r') as f:
            data = json.load(f)
        
        if not 'occupies_no_train' in data:
            skipped.append((subset, 'please rerun concrete. outdated (occupies_no_train)'))
            continue

        xs_concrete_train = data['occupies']
        ys_concrete_train = data['metrics']
        xs_concrete_no_train = data['occupies_no_train']
        ys_concrete_no_train = data['metrics_no_train']
        metric_name = data['metric_name']
        concrete_p_logits = data['p_logits']

        # load ltp
        if not os.path.exists(ltp_data_path):
            skipped.append((subset, 'please run ltp'))
            continue
        
        with open(ltp_data_path, 'r') as f:
            data = json.load(f)

        if not 'max_test_occupies' in data:
            skipped.append((subset, 'please rerun ltp. outdated (max_test_occupies)'))
            continue
        
        xs_ltp = data['max_test_occupies']
        ys_ltp = data['max_test_metrics']
        flat_ltp = [(xs_ltp[i], ys_ltp[i])for i in range(len(xs_ltp))]
        flat_ltp = list(sorted(flat_ltp, key=lambda x: x[0]))
        xs_ltp = [it[0] for it in flat_ltp]
        ys_ltp = [it[1] for it in flat_ltp]
        ltp_lambdas = data['lambdas']
        ltp_temperatures = data['temperatures']

        all_xs =\
            xs_absatt +\
            xs_forward +\
            xs_sparse +\
            xs_concrete_no_train +\
            xs_concrete_train +\
            xs_ltp
        xs_bert = [min(all_xs), max(all_xs)]

        # save data
        data = {
            'xs_absatt': xs_absatt,
            'ys_absatt': ys_absatt,
            'xs_forward': xs_forward,
            'ys_forward': ys_forward,
            'xs_sparse': xs_sparse,
            'ys_sparse': ys_sparse,
            'xs_bert': xs_bert,
            'ys_bert': ys_bert,
            'xs_concrete_train': xs_concrete_train,
            'ys_concrete_train': ys_concrete_train,
            'xs_concrete_no_train': xs_concrete_no_train,
            'ys_concrete_no_train': ys_concrete_no_train,
            'xs_ltp': xs_ltp,
            'ys_ltp': ys_ltp,
            'metric_name': metric_name,
            'concrete_p_logits': concrete_p_logits,
            'ltp_lambdas': ltp_lambdas,
            'ltp_temperatures': ltp_temperatures,
            'ks': ks,
        }
        with open(plot_name + '.json', 'w') as f:
            json.dump(data, f)
        
        # render plot
        metric_display_name = metric_to_name[metric_name]
        y_scale = metric_to_scaler[metric_name]
        ys_absatt = scale(ys_absatt, y_scale)
        ys_sparse = scale(ys_sparse, y_scale)
        ys_concrete_train = scale(ys_concrete_train, y_scale)
        ys_concrete_no_train = scale(ys_concrete_no_train, y_scale)
        ys_ltp = scale(ys_ltp, y_scale)
        ys_forward = scale(ys_forward, y_scale)
        ys_bert = scale(ys_bert, y_scale)

        x_scale = 100
        xs_absatt = scale(xs_absatt, x_scale)
        xs_sparse = scale(xs_sparse, x_scale)
        xs_concrete_train = scale(xs_concrete_train, x_scale)
        xs_concrete_no_train = scale(xs_concrete_no_train, x_scale)
        xs_ltp = scale(xs_ltp, x_scale)
        xs_forward = scale(xs_forward, x_scale)
        xs_bert = scale(xs_bert, x_scale)

        plt.clf()
        plt.plot(xs_sparse, ys_sparse, marker='o', label='STTBTA (Approx. Att.)', linewidth=1.2)
        plt.plot(xs_absatt, ys_absatt, marker='o', label='STTBTA (Actual Att.)', linewidth=1.2)
        plt.plot(xs_concrete_train, ys_concrete_train, marker='^', label='STTBTA (Concrete, with train)', linewidth=1.2)
        plt.plot(xs_concrete_no_train, ys_concrete_no_train, marker='^', label='STTBTA (Concrete, w/o train)', linewidth=1.2)
        plt.plot(xs_ltp, ys_ltp, marker='x', label='LTP (Best valid.)', linewidth=1.2, color='gray', linestyle='--')
        plt.plot(xs_forward, ys_forward, marker='x', label='Manual Top-k', linewidth=1.2, color='black', linestyle='--')
        plt.plot(xs_bert, ys_bert, linestyle=':', label='BERT$_{BASE}$', color='skyblue', zorder=-99)
        plt.grid(True)
        plt.xlabel('Average Keep Token Ratio (%)')
        plt.ylabel(metric_display_name)
        plt.legend()
        plt.title(f'{subset_to_name[subset]}', fontsize=12)
        plt.savefig(plot_name+'.png', dpi=320)

        print(f'{subset} is processed')

    if len(skipped) > 0:
        print('-- skipped subsets --')
        print(*skipped, sep='\n')
    else:
        print('All subsets are processed!')

if __name__ == '__main__':
    main()