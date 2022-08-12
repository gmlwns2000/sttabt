import os, pickle, json
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from main.plot.constants import *

plt.style.use(PLT_STYLE)

"""
Example --

data[('cola', 0.05)] = {
    'occupy': 0.12603720492206785,
    'score_sparse': 0.1766619570140954,
    'flops_sparse': 0.8340102187489314,
    'occupy_forward': 0.17344440644499617,
    'score_forward': 0.07020916571551028,
    'flops_forward': 0.7350509942306107,
    'metric': 'matthews_correlation',
    'score_sparse_approx': 0.0820815354201135,
    'occupy_approx': 0.11403657924786666,
    'flops_approx': 0.7666914154334941
}
"""
def render(data, header):
    subsets = set()
    kxs = set()
    for key in data.keys():
        subsets.add(key[0])
        if key[1] != 'bert':
            kxs.add(key[1])
    kxs = list(sorted(kxs))

    combined_fig = plt.figure(figsize=(21,7))
    
    for subset in subsets:
        metric = data[(subset, kxs[0])]['metric']
        y_scale = METRIC_TO_SCALER[metric]
        xs_forward = [data[(subset, kx)]['flops_forward'] for kx in kxs]
        ys_forward = [data[(subset, kx)]['score_forward'] for kx in kxs]
        ys_forward = scale(ys_forward, y_scale)
        xs_approx = [data[(subset, kx)]['flops_approx'] for kx in kxs]
        ys_approx = [data[(subset, kx)]['score_sparse_approx'] for kx in kxs]
        ys_approx = scale(ys_approx, y_scale)
        xs_absatt = [data[(subset, kx)]['flops_sparse'] for kx in kxs]
        ys_absatt = [data[(subset, kx)]['score_sparse'] for kx in kxs]
        ys_absatt = scale(ys_absatt, y_scale)
        xs_bert = xs_forward + xs_approx + xs_absatt
        xs_bert = [min(xs_bert), max(xs_bert)]
        ys_bert = [data[(subset, 'bert')]['score_bert'],]*2
        ys_bert = scale(ys_bert, y_scale)

        plt.clf()

        plt.figure(figsize=(5, 4))
        plt.plot(
            xs_approx, ys_approx, 
            label=STR_STTABT_APPROX, color=COLOR_STTABT_APPROX,
            marker='o', zorder=10
        )
        plt.plot(
            xs_absatt, ys_absatt, 
            label=STR_STTABT_ABSATT, color=COLOR_STTABT_ABSATT,
            marker='o', 
        )
        plt.plot(
            xs_forward, ys_forward, 
            color=COLOR_MANUAL_TOPK, label=STR_MANUAL_TOPK,
            marker='x', linestyle='--', 
        )
        plt.plot(
            xs_bert, ys_bert, 
            label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
            linestyle=':', zorder=-99
        )
        plt.grid(True)
        plt.xlabel(STR_GFLOPS)
        plt.ylabel(METRIC_TO_NAME[metric])
        plt.legend()
        plt.title(SUBSET_TO_NAME[subset], fontsize=12)

        plot_name = f'./saves_plot/approx-glue-{header}-{subset}-flops'
        plt.savefig(plot_name+'.svg')

        with open(plot_name + '.json', 'w') as f:
            json.dump({
                'xs_forward': xs_forward,
                'ys_forward': ys_forward,
                'xs_approx': xs_approx,
                'ys_approx': ys_approx,
                'xs_absatt': xs_absatt,
                'ys_absatt': ys_absatt,
                'xs_bert': xs_bert,
                'ys_bert': ys_bert,
                'subset': subset,
                'kxs': kxs,
            }, f)
        
        if GLUE_SUBSETS.index(subset) < 5:
            ax = combined_fig.add_subplot(2, 10, 
                (GLUE_SUBSETS.index(subset)*2 + 1, GLUE_SUBSETS.index(subset)*2 + 2)
            )
        else:
            ax = combined_fig.add_subplot(2, 10, 
                (GLUE_SUBSETS.index(subset)*2 + 2, GLUE_SUBSETS.index(subset)*2 + 3)
            )
        ax.plot(
            xs_approx, ys_approx, 
            label=STR_STTABT_APPROX, color=COLOR_STTABT_APPROX,
            marker='o', zorder=10
        )
        ax.plot(
            xs_absatt, ys_absatt, 
            label=STR_STTABT_ABSATT, color=COLOR_STTABT_ABSATT,
            marker='o', 
        )
        ax.plot(
            xs_forward, ys_forward, 
            color=COLOR_MANUAL_TOPK, label=STR_MANUAL_TOPK,
            marker='x', linestyle='--', 
        )
        ax.plot(
            xs_bert, ys_bert, 
            label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
            linestyle=':', zorder=-99
        )
        ax.grid(True)
        ax.set_xlabel(STR_GFLOPS)
        ax.set_ylabel(METRIC_TO_NAME[metric])
        #ax.legend()
        ax.set_title(SUBSET_TO_NAME[subset], fontsize=16)
    
    handles, labels = ax.get_legend_handles_labels()
    legend = combined_fig.legend(handles, labels, loc='lower center', fontsize=16, ncol = 4)
    combined_fig.tight_layout()
    combined_fig.subplots_adjust(bottom=0.15)
    combined_fig.savefig(
        f'./saves_plot/approx-glue-{header}-all-flops.png',
        bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
    )
    combined_fig.savefig(
        f'./saves_plot/approx-glue-{header}-all-flops.svg',
        bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320
    )

def main(path, header):
    if not os.path.exists(path):
        print('Main: Path is not exists,', path)
        return
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    try:
        render(data, header)
        print('Main: Rendered', path)
    except Exception as ex:
        import traceback
        print('Main: Error while render')
        traceback.print_exc()
        print('Main:', ex)
        print('Main:', path)

if __name__ == '__main__':
    f4_pickle_path = "./saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle"
    f8_pickle_path = "./saves_plot/[F8-PREWIKI.v2]glue_benchmark_accum_absatt.pickle"
    main(f4_pickle_path, 'f4')
    main(f8_pickle_path, 'f8')