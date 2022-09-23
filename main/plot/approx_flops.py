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
def render(data_f4, data_f8, header):
    subsets = set()
    kxs = set()
    for key in data_f4.keys():
        subsets.add(key[0])
        if key[1] != 'bert':
            kxs.add(key[1])
    kxs = list(sorted(kxs))

    layout = '3x3'
    if layout == '3x3':
        combined_fig = plt.figure(figsize=(12,9.5))
    else:
        combined_fig = plt.figure(figsize=(21,7))
    
    data = []
    for subset in subsets:
        metric = data_f4[(subset, kxs[0])]['metric']
        y_scale = METRIC_TO_SCALER[metric]
        xs_forward = [data_f4[(subset, kx)]['flops_forward'] for kx in kxs]
        ys_forward = [data_f4[(subset, kx)]['score_forward'] for kx in kxs]
        ys_forward = scale(ys_forward, y_scale)
        xs_approx = [data_f4[(subset, kx)]['flops_approx'] for kx in kxs]
        ys_approx = [data_f4[(subset, kx)]['score_sparse_approx'] for kx in kxs]
        ys_approx = scale(ys_approx, y_scale)
        xs_absatt = [data_f4[(subset, kx)]['flops_sparse'] for kx in kxs]
        ys_absatt = [data_f4[(subset, kx)]['score_sparse'] for kx in kxs]
        ys_absatt = scale(ys_absatt, y_scale)
        xs_approx8 = [data_f8[(subset, kx)]['flops_approx'] for kx in kxs]
        ys_approx8 = [data_f8[(subset, kx)]['score_sparse_approx'] for kx in kxs]
        ys_approx8 = scale(ys_approx8, y_scale)
        xs_absatt8 = [data_f8[(subset, kx)]['flops_sparse'] for kx in kxs]
        ys_absatt8 = [data_f8[(subset, kx)]['score_sparse'] for kx in kxs]
        ys_absatt8 = scale(ys_absatt8, y_scale)
        xs_bert = xs_forward + xs_approx + xs_absatt + xs_approx8 + xs_absatt8
        xs_bert = [min(xs_bert), max(xs_bert)]
        ys_bert = [data_f4[(subset, 'bert')]['score_bert'],]*2
        ys_bert = scale(ys_bert, y_scale)

        plot_name = f'./saves_plot/approx-glue-{header}-{subset}-flops'
        with open(plot_name + '.json', 'w') as f:
            dump = {
                'xs_forward': xs_forward,
                'ys_forward': ys_forward,
                'xs_approx': xs_approx,
                'ys_approx': ys_approx,
                'xs_absatt': xs_absatt,
                'ys_absatt': ys_absatt,
                'xs_approx8': xs_approx8,
                'ys_approx8': ys_approx8,
                'xs_absatt8': xs_absatt8,
                'ys_absatt8': ys_absatt8,
                'xs_bert': xs_bert,
                'ys_bert': ys_bert,
                'subset': subset,
                'kxs': kxs,
            }
            data.append(dump)
            json.dump(dump, f)
        
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
            xs_approx8, ys_approx8, 
            label=STR_STTABT_APPROX_F8, color=COLOR_STTABT_APPROX,
            marker='o', zorder=100
        )
        ax.plot(
            xs_absatt8, ys_absatt8, 
            label=STR_STTABT_ABSATT_F8, color=COLOR_STTABT_ABSATT,
            marker='o', 
        )
        ax.plot(
            xs_approx, ys_approx, 
            label=STR_STTABT_APPROX_F4, color=COLOR_STTABT_APPROX_F4,
            marker='.', zorder=10, linestyle='--'
        )
        ax.plot(
            xs_absatt, ys_absatt, 
            label=STR_STTABT_ABSATT_F4, color=COLOR_STTABT_ABSATT_F4,
            marker='.', linestyle='--'
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
        ymin, ymax = ax.get_ylim()
        subset_to_ylim = {
            'mnli': 0.3,
            'qnli': 0.3,
            'mrpc': 0.99,
            'cola': 0.33,
            'qqp': 0.4,
            'wnli': 2.0,
            'stsb': 0.1,
            'sst2': 0.5,
            'rte': 0.72,
        }
        ax.set_ylim(ymax - (ymax-ymin)*subset_to_ylim.get(subset, 0.7), ymax)
        ymin, ymax = ax.get_ylim()
        subset_to_ylim_upper = {
            'stsb': 0.66,
            'wnli': 1.7
        }
        ax.set_ylim(ymin, ymin + (ymax-ymin)*subset_to_ylim_upper.get(subset, 1.0))
        xmin, xmax = ax.get_xlim()
        subset_to_xlim = {
            'stsb': 0.8,
            'qnli': 0.95,
            'cola': 0.9,
        }
        ax.set_xlim(xmax - (xmax-xmin)*subset_to_xlim.get(subset, 1.0), xmax)
        ax.grid(True)
        ax.set_xlabel(STR_GFLOPS)
        ax.set_ylabel(METRIC_TO_NAME[metric])
        #ax.legend()
        ax.set_title(SUBSET_TO_NAME[subset], fontsize=16)
    
    if layout == '3x3':
        handles, labels = ax.get_legend_handles_labels()
        legend = combined_fig.legend(handles, labels, loc='lower center', fontsize=12, ncol = 4)
        combined_fig.tight_layout()
        combined_fig.subplots_adjust(bottom=0.12)
        combined_fig.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops.png',
            bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320, pad_inches=0.05
        )
        combined_fig.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops.svg',
            bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320, pad_inches=0.05
        )
        combined_fig.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops.pdf',
            bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320, pad_inches=0.05
        )
    else:
        handles, labels = ax.get_legend_handles_labels()
        legend = combined_fig.legend(handles, labels, loc='lower center', fontsize=16, ncol = 4)
        combined_fig.tight_layout()
        combined_fig.subplots_adjust(bottom=0.19)
        combined_fig.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops.png',
            bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320, pad_inches=0.05
        )
        combined_fig.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops.svg',
            bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320, pad_inches=0.05
        )
        combined_fig.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops.pdf',
            bbox_extra_artists=(legend, ), bbox_inches='tight', dpi=320, pad_inches=0.05
        )

    #reduced graph
    """
    data = [{
        'xs_forward': xs_forward,
        'ys_forward': ys_forward,
        'xs_approx': xs_approx,
        'ys_approx': ys_approx,
        'xs_absatt': xs_absatt,
        'ys_absatt': ys_absatt,
        'xs_approx8': xs_approx8,
        'ys_approx8': ys_approx8,
        'xs_absatt8': xs_absatt8,
        'ys_absatt8': ys_absatt8,
        'xs_bert': xs_bert,
        'ys_bert': ys_bert,
        'subset': subset,
        'kxs': kxs,
    }]
    """
    #reduce graphs
    def interpolate_point(xs, ys, tx):
        if min(xs) >= tx:
            return
        if max(xs) <= tx:
            return
        for i, x in enumerate(xs):
            if x > tx:
                if i == 0: return
                prex = xs[i-1]
                prey = ys[i-1]
                y = ys[i]
                ty = (1-(tx-prex)/(x-prex)) * prey + (tx-prex)/(x-prex) * y
                return ty
    line_names = 'forward approx absatt approx8 absatt8'.split()
    #change xs from GFLOPs to Rel. FLOPs
    # change ys from METRIC to Rel.
    with open('saves_plot/glue_bert_flops.json', 'r') as f:
        flops_data = json.load(f)
    for entry in data:
        subset = entry['subset']
        bert_flops = flops_data[subset]
        bert_metric = entry['ys_bert'][0]
        #sort & scale
        for line_name in line_names:
            xs = scale(entry[f'xs_{line_name}'], 100/bert_flops)
            ys = scale(entry[f'ys_{line_name}'], 100/bert_metric)
            ds = sorted(zip(xs, ys), key=lambda it: it[0])
            xs = [d[0] for d in ds]
            ys = [d[1] for d in ds]
            entry[f'xs_{line_name}'] = xs
            entry[f'ys_{line_name}'] = ys
    # interpolate and average data points
    lines = { name:[] for name in line_names }
    for entry in data:
        for line_name in line_names:
            xs = entry[f'xs_{line_name}']
            ys = entry[f'ys_{line_name}']
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                y_count = 1
                for entry_ in data:
                    if entry_['subset'] == entry['subset']:
                        continue
                    ty = interpolate_point(
                        entry_[f'xs_{line_name}'], entry_[f'ys_{line_name}'], x
                    )
                    if ty is not None:
                        y += ty
                        y_count += 1
                y /= y_count
                lines[line_name].append((x, y))
    for k in lines.keys():
        lines[k] = sorted(lines[k], key=lambda it: it[0])
    #render
    xs_approx8, ys_approx8 = [[pt[i] for pt in lines['approx8']] for i in range(2)]
    xs_absatt8, ys_absatt8 = [[pt[i] for pt in lines['absatt8']] for i in range(2)]
    xs_approx, ys_approx = [[pt[i] for pt in lines['approx']] for i in range(2)]
    xs_absatt, ys_absatt = [[pt[i] for pt in lines['absatt']] for i in range(2)]
    xs_forward, ys_forward = [[pt[i] for pt in lines['forward']] for i in range(2)]
    xs_bert = xs_approx8 + xs_absatt8 + xs_approx + xs_absatt + xs_forward
    xs_bert = [min(xs_bert), max(xs_bert)]
    ys_bert = [100.0, 100.0]
    plt.clf()
    fig_scale = 0.7
    plt.figure(figsize=(6.4*fig_scale, 4.8*fig_scale))
    plt.plot(
        xs_approx8, ys_approx8, 
        label=STR_STTABT_APPROX_F8, color=COLOR_STTABT_APPROX,
        marker='', zorder=100
    )
    plt.plot(
        xs_absatt8, ys_absatt8, 
        label=STR_STTABT_ABSATT_F8, color=COLOR_STTABT_ABSATT,
        marker='', 
    )
    plt.plot(
        xs_approx, ys_approx, 
        label=STR_STTABT_APPROX_F4, color=COLOR_STTABT_APPROX_F4,
        marker='', zorder=10, linestyle='--'
    )
    plt.plot(
        xs_absatt, ys_absatt, 
        label=STR_STTABT_ABSATT_F4, color=COLOR_STTABT_ABSATT_F4,
        marker='', linestyle='--'
    )
    plt.plot(
        xs_forward, ys_forward, 
        color=COLOR_MANUAL_TOPK, label=STR_MANUAL_TOPK,
        marker='', linestyle='--', 
    )
    plt.plot(
        xs_bert, ys_bert, 
        label=STR_BERT_BASE, color=COLOR_BERT_BASE, 
        linestyle=':', zorder=-99
    )
    ymin, ymax = plt.ylim()
    plt.ylim(ymax - (ymax-ymin) * 0.3, ymax)
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymin + (ymax-ymin)*1.0)
    xmin, xmax = plt.xlim()
    plt.xlim(xmax - (xmax-xmin)*0.9, xmax)
    plt.grid(True)
    plt.xlabel('Relative FLOPs (%)')
    plt.ylabel('Relative Metric (%)')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend(prop={'size':8})
    #ax.legend()
    plt.title('Average GLUE', fontsize=12)
    if layout == '3x3':
        plt.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops-reduced.png',
            bbox_inches='tight', dpi=320, pad_inches=0.05
        )
        plt.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops-reduced.svg',
            bbox_inches='tight', dpi=320, pad_inches=0.05
        )
        plt.savefig(
            f'./saves_plot/approx-glue-{header}-all-flops-reduced.pdf',
            bbox_inches='tight', dpi=320, pad_inches=0.05
        )
    else: raise Exception()

def main(path_f4, path_f8, header):
    if not os.path.exists(path_f4):
        print('Main: Path is not exists,', path_f4)
        return
    if not os.path.exists(path_f8):
        print('Main: Path is not exists,', path_f8)
        return
    
    with open(path_f4, 'rb') as f:
        data_f4 = pickle.load(f)
    with open(path_f8, 'rb') as f:
        data_f8 = pickle.load(f)
    
    try:
        render(data_f4, data_f8, header)
        print('Main: Rendered', path_f4, path_f8)
    except Exception as ex:
        import traceback
        print('Main: Error while render')
        traceback.print_exc()
        print('Main:', ex)

if __name__ == '__main__':
    f4_pickle_path = "./saves_plot/[F4-PREWIKI.v2]glue_benchmark_accum_absatt.pickle"
    f8_pickle_path = "./saves_plot/[F8-PREWIKI.v2]glue_benchmark_accum_absatt.pickle"
    main(f4_pickle_path, f8_pickle_path, 'all')