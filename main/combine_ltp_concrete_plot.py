import json, math, random, pickle, os
from matplotlib import pyplot as plt

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
            print(f'concrete result is not exists ({subset})')
            continue
        
        with open(concrete_data_path, 'r') as f:
            data = json.load(f)
        
        xs_concrete_train = data['occupies']
        ys_concrete_train = data['metrics']
        xs_concrete_no_train = data['occupies_no_train']
        ys_concrete_no_train = data['metrics_no_train']
        metric_name = data['metric_name']
        concrete_p_logits = data['p_logits']

        # load ltp
        if not os.path.exists(ltp_data_path):
            skipped.append((subset, 'please run ltp'))
            print(f'ltp result is not exists ({subset})')
            continue
        
        with open(ltp_data_path, 'r') as f:
            data = json.load(f)

        xs_ltp = data['max_test_occupies']
        ys_ltp = data['max_test_metrics']
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
        plt.clf()
        plt.savefig(plot_name+'.png', dpi=320)

    print(*skipped, sep='\n')

if __name__ == '__main__':
    main()