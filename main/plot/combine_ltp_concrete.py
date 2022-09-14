import json, math, random, pickle, os
from matplotlib import pyplot as plt
plt.style.use('seaborn-bright')

from main.plot.constants import *

def main():
    baseline_data_path = 'saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle'
    subsets = GLUE_SUBSETS
    skipped = []

    layout = '3x3'
    if layout == '3x3':
        combined_fig = plt.figure(figsize=(12,9.5))
    else:
        combined_fig = plt.figure(figsize=(21,7))
    plt.figure()
    ax = None

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
            json.dump(data, f, indent=2)
        
        # render plot
        metric_display_name = METRIC_TO_NAME[metric_name]
        y_scale = METRIC_TO_SCALER[metric_name]
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

        # plt.clf()
        # plt.plot(
        #     xs_sparse, ys_sparse, 
        #     label=STR_STTABT_APPROX, color=COLOR_STTABT_APPROX, 
        #     marker='o', linewidth=1.2, zorder=10
        # )
        # plt.plot(
        #     xs_absatt, ys_absatt, 
        #     label=STR_STTABT_ABSATT, color=COLOR_STTABT_ABSATT,
        #     marker='o', linewidth=1.2, 
        # )
        # plt.plot(
        #     xs_concrete_train, ys_concrete_train, 
        #     label=STR_STTABT_CONCRETE_WITH_TRAIN, color=COLOR_STTABT_CONCRETE_WITH_TRAIN, 
        #     marker='^', linewidth=1.2, zorder=10
        # )
        # plt.plot(
        #     xs_concrete_no_train, ys_concrete_no_train, 
        #     label=STR_STTABT_CONCRETE_WO_TRAIN, color=COLOR_STTABT_CONCRETE_WO_TRAIN,
        #     marker='^', linewidth=1.2, 
        # )
        # plt.plot(
        #     xs_ltp, ys_ltp, 
        #     label=STR_LTP_BEST_VALID, color=COLOR_LTP_BEST_VALID, 
        #     marker='x', linewidth=1.2, linestyle='--'
        # )
        # plt.plot(
        #     xs_forward, ys_forward, 
        #     label=STR_MANUAL_TOPK, color=COLOR_MANUAL_TOPK, 
        #     marker='x', linewidth=1.2, linestyle='--'
        # )
        # plt.plot(
        #     xs_bert, ys_bert, 
        #     label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
        #     linestyle=':', zorder=-99
        # )
        # plt.grid(True)
        # plt.xlabel(STR_AVERAGE_KEEP_TOKEN_RATIO)
        # plt.ylabel(metric_display_name)
        # plt.legend()
        # plt.title(f'{SUBSET_TO_NAME[subset]}', fontsize=12)
        # plt.savefig(plot_name+'.svg', dpi=320)

        def bert_xs(*xss):
            xss = sum(xss, start=[])
            return [min(xss), max(xss)]

        # plt.clf()
        # plt.plot(
        #     xs_sparse, ys_sparse, 
        #     label=STR_STTABT_APPROX, color=COLOR_STTABT_APPROX, 
        #     marker='o', linewidth=1.2, zorder=10
        # )
        # plt.plot(
        #     xs_absatt, ys_absatt, 
        #     label=STR_STTABT_ABSATT, color=COLOR_STTABT_ABSATT,
        #     marker='o', linewidth=1.2, 
        # )
        # plt.plot(
        #     xs_concrete_no_train, ys_concrete_no_train, 
        #     label=STR_STTABT_CONCRETE_WO_TRAIN, color=COLOR_STTABT_CONCRETE_WO_TRAIN, 
        #     marker='^', linewidth=1.2, 
        # )
        # plt.plot(
        #     xs_forward, ys_forward, 
        #     label=STR_MANUAL_TOPK, color=COLOR_MANUAL_TOPK, 
        #     marker='x', linewidth=1.2, linestyle='--'
        # )
        # plt.plot(
        #     bert_xs(xs_sparse, xs_absatt, xs_concrete_no_train, xs_forward), ys_bert, 
        #     label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
        #     linestyle=':', zorder=-99
        # )
        # plt.grid(True)
        # plt.xlabel(STR_AVERAGE_KEEP_TOKEN_RATIO)
        # plt.ylabel(metric_display_name)
        # plt.legend()
        # plt.title(f'{SUBSET_TO_NAME[subset]}', fontsize=12)
        # plt.savefig(plot_name+'-no-train.svg', dpi=320)

        # plt.clf()
        # plt.plot(
        #     xs_concrete_train, ys_concrete_train, 
        #     label=STR_STTABT_CONCRETE_WITH_TRAIN, color=COLOR_STTABT_CONCRETE_WITH_TRAIN, 
        #     marker='^', linewidth=1.2, zorder=10
        # )
        # plt.plot(
        #     xs_ltp, ys_ltp, 
        #     label=STR_LTP_BEST_VALID, color=COLOR_LTP_BEST_VALID, 
        #     linewidth=1.2, marker='x', linestyle='--'
        # )
        # plt.plot(
        #     bert_xs(xs_concrete_train, xs_ltp), ys_bert, 
        #     label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
        #     linestyle=':', zorder=-99
        # )
        # plt.grid(True)
        # plt.xlabel(STR_AVERAGE_KEEP_TOKEN_RATIO)
        # plt.ylabel(metric_display_name)
        # plt.legend()
        # plt.title(f'{SUBSET_TO_NAME[subset]}', fontsize=12)
        # plt.savefig(plot_name+'-train.svg', dpi=320)
        
        if layout == '3x3':
            ax = combined_fig.add_subplot(3, 3, 
                GLUE_SUBSETS.index(subset)+1
            )
        else:
            if GLUE_SUBSETS.index(subset) < 5:
                ax = combined_fig.add_subplot(2, 10, 
                    (GLUE_SUBSETS.index(subset)*2 + 1, GLUE_SUBSETS.index(subset)*2 + 2)
                )
            else:
                ax = combined_fig.add_subplot(2, 10, 
                    (GLUE_SUBSETS.index(subset)*2 + 2, GLUE_SUBSETS.index(subset)*2 + 3)
                )
        ax.plot(
            xs_sparse, ys_sparse, 
            label=STR_STTABT_APPROX, color=COLOR_STTABT_APPROX, 
            marker='o', zorder=10
        )
        ax.plot(
            xs_absatt, ys_absatt, 
            label=STR_STTABT_ABSATT, color=COLOR_STTABT_ABSATT,
            marker='o', 
        )
        ax.plot(
            xs_concrete_train, ys_concrete_train, 
            label=STR_STTABT_CONCRETE_WITH_TRAIN, color=COLOR_STTABT_CONCRETE_WITH_TRAIN, 
            marker='^', zorder=10
        )
        ax.plot(
            xs_concrete_no_train, ys_concrete_no_train, 
            label=STR_STTABT_CONCRETE_WO_TRAIN, color=COLOR_STTABT_CONCRETE_WO_TRAIN,
            marker='^', 
        )
        ax.plot(
            xs_ltp, ys_ltp, 
            label=STR_LTP_BEST_VALID, color=COLOR_LTP_BEST_VALID, 
            marker='x', linestyle='--'
        )
        ax.plot(
            xs_forward, ys_forward, 
            label=STR_MANUAL_TOPK, color=COLOR_MANUAL_TOPK, 
            marker='x', linestyle='--'
        )
        ax.plot(
            xs_bert, ys_bert, 
            label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
            linestyle=':', zorder=-99
        )
        ymin, ymax = ax.get_ylim()
        subset_to_ylim = {
            'qnli': 0.6,
            'mrpc': 0.35,
            'cola': 0.85,
            'qqp': 0.4,
            'wnli': 0.2,
            'stsb': 0.2,
            'sst2': 0.3,
            'rte': 0.72,
        }
        ax.set_ylim(ymax - (ymax-ymin)*subset_to_ylim.get(subset, 0.7), ymax)
        ymin, ymax = ax.get_ylim()
        subset_to_ylim = {
            'stsb': 0.95,
        }
        ax.set_ylim(ymin, ymin + (ymax-ymin)*subset_to_ylim.get(subset, 1.0))
        ax.grid(True)
        ax.set_xlabel(STR_AVERAGE_KEEP_TOKEN_RATIO)
        ax.set_ylabel(metric_display_name)
        ax.set_title(f'{SUBSET_TO_NAME[subset]}', fontsize=12)

        print(f'{subset} is processed')
    
    if ax is None:
        print('None of subset are processed. skip render all plot')
    else:
        if layout == '3x3':
            handles, labels = ax.get_legend_handles_labels()
            legend = combined_fig.legend(handles, labels, loc='lower center', fontsize=12, ncol = 4)
            combined_fig.tight_layout()
            combined_fig.subplots_adjust(bottom=0.13)
            combined_fig.savefig(
                f'saves_plot/combined-glue-all.png',
                bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
            )
            combined_fig.savefig(
                f'saves_plot/combined-glue-all.svg',
                bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
            )
            combined_fig.savefig(
                f'saves_plot/combined-glue-all.pdf',
                bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
            )
        else:
            handles, labels = ax.get_legend_handles_labels()
            legend = combined_fig.legend(handles, labels, loc='lower center', fontsize=16, ncol = 4)
            combined_fig.tight_layout()
            combined_fig.subplots_adjust(bottom=0.2)
            combined_fig.savefig(
                f'saves_plot/combined-glue-all.png',
                bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
            )
            combined_fig.savefig(
                f'saves_plot/combined-glue-all.svg',
                bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
            )
            combined_fig.savefig(
                f'saves_plot/combined-glue-all.pdf',
                bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
            )

    if len(skipped) > 0:
        print('-- skipped subsets --')
        print(*skipped, sep='\n')
    else:
        print('All subsets are processed!')

if __name__ == '__main__':
    main()