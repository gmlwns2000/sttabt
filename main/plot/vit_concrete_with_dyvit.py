"""
this script will combine the result between utils.dyvit_occupy and main.vit_concrete_end2end
"""
import json, os
from matplotlib import pyplot as plt
plt.style.use('seaborn-bright')
from main.plot.constants import *

PLOT_NAME = './saves_plot/vit-concrete-imnet'
DATASET='imnet'
#TODO! support other datasets
assert DATASET == 'imnet'

def load_dyvit():
    from utils import dyvit_occupy
    results = dyvit_occupy.DYVIT_RESULTS
    pts = list(sorted([(r['occupy'], r['accuracy']) for r in results], key=lambda it: it[0]))
    return ([p[i] for p in pts] for i in range(2))

def load_concrete(factor=4, p_logits=[-3, -2, -1.5, -1.25, -1, -0.5, 0.0, 0.5, 1.0], epochs=20, warmup_epochs=14):
    #saves/dyvit-concrete-f{factor}-{plogit}-nohard-e{epochs}-we{warmup_epochs}/log.txt
    #get last line of it
    pts = []
    for p in p_logits:
        log_path = f'./saves/dyvit-concrete-f{factor}-{p}-nohard-e{epochs}-we{warmup_epochs}/log.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < epochs:
                    print(f'LoadConcrete: skip, train is incomplete at epochs {len(lines)}', log_path)
                    continue
                last_checkpoint = lines[-1]
                result = json.loads(last_checkpoint)
                pts.append((
                    result['test_hard_occupy'], 
                    result['test_acc1'], 
                    result['test_hard_occupy_ema'], 
                    result['test_acc1_ema']
                ))
        else:
            print('LoadConcrete: skip, file no exists', log_path)
    return ([p[i] for p in pts] for i in range(4))

def load_approx(factor=4):
    json_path = './saves_plot/vit-approx-base.json'
    assert factor == 4

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    assert data['factor'] == factor

    ys_base = data['metric_baseline']
    
    xs_approx = data['occupies_sparse']
    ys_approx = data['metrics_sparse']
    
    xs_absatt = data['occupies_absatt']
    ys_absatt = data['metrics_absatt']

    xs_forward = data['occupies_forward']
    ys_forward = data['metrics_forward']

    # should return 0~100 for metric
    yscale = 100
    ys_approx = scale(ys_approx, yscale)
    ys_absatt = scale(ys_absatt, yscale)
    ys_forward = scale(ys_forward, yscale)
    ys_base = ys_base * yscale

    return (
        xs_approx, ys_approx,
        xs_absatt, ys_absatt,
        xs_forward, ys_forward,
        ys_base,
    )

def main(factor=4):
    xs_dyvit, ys_dyvit = load_dyvit()
    (
        xs_concrete, ys_concrete, 
        xs_concrete_ema, ys_concrete_ema
    ) = load_concrete(factor=factor)
    (
        xs_approx, ys_approx, 
        xs_absatt, ys_absatt, 
        xs_forward, ys_forward, 
        ys_base
    ) = load_approx(factor=factor)
    ys_base = [ys_base, ] * 2
    xs_base = xs_dyvit + xs_concrete + xs_concrete_ema + xs_approx + xs_absatt + xs_forward
    xs_base = [min(xs_base), max(xs_base)]

    json_path = f'{PLOT_NAME}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'xs_dyvit': xs_dyvit,
            'ys_dyvit': ys_dyvit,
            'xs_concrete': xs_concrete,
            'ys_concrete': ys_concrete,
            'xs_concrete_ema': xs_concrete_ema,
            'ys_concrete_ema': ys_concrete_ema,
            'xs_base': xs_base,
            'ys_base': ys_base,
            'xs_approx': xs_approx,
            'ys_approx': ys_approx,
            'xs_absatt': xs_absatt,
            'ys_absatt': ys_absatt,
            'xs_forward': xs_forward,
            'ys_forward': ys_forward,
        }, f, indent=2)

    plt.clf()

    xscale = 100
    xs_dyvit = scale(xs_dyvit, xscale)
    xs_approx = scale(xs_approx, xscale)
    xs_absatt = scale(xs_absatt, xscale)
    xs_forward = scale(xs_forward, xscale)
    xs_concrete = scale(xs_concrete, xscale)
    xs_concrete_ema = scale(xs_concrete_ema, xscale)
    xs_base = scale(xs_base, xscale)
    
    plt.plot(
        xs_concrete, ys_concrete,
        label=STR_STTABT_CONCRETE_WITH_TRAIN, color=COLOR_STTABT_CONCRETE_WITH_TRAIN,
        marker='^', linestyle='-', linewidth=1.2, zorder=99,
    )
    plt.plot(
        xs_concrete_ema, ys_concrete_ema,
        label=STR_STTABT_CONCRETE_WITH_TRAIN_EMA, color=COLOR_STTABT_CONCRETE_WITH_TRAIN,
        marker='^', linestyle='--', linewidth=1.2,
    )
    plt.plot(
        xs_dyvit, ys_dyvit,
        label=STR_DYNAMIC_VIT, color=COLOR_LTP_BEST_VALID,
        marker='x', linestyle='--', linewidth=1.2, zorder=1,
    )
    plt.plot(
        xs_base, ys_base, 
        label=STR_DEIT_SMALL, color=COLOR_BERT_BASE, 
        linestyle=':', zorder=-99,
    )

    #backup ylim
    y_bot, y_top = plt.ylim()

    plt.plot(
        xs_approx, ys_approx,
        label=STR_STTABT_APPROX, color=COLOR_STTABT_APPROX,
        marker='o', linewidth=1.2, zorder=9,
    )
    plt.plot(
        xs_absatt, ys_absatt,
        label=STR_STTABT_ABSATT, color=COLOR_STTABT_ABSATT,
        marker='o', linewidth=1.2, zorder=10,
    )
    plt.plot(
        xs_forward, ys_forward,
        label=STR_MANUAL_TOPK, color=COLOR_MANUAL_TOPK,
        marker='x', linewidth=1.2, linestyle='--', zorder=-10,
    )

    #restore ylim
    plt.ylim((y_bot, y_top))
    
    plt.grid(True)
    plt.xlabel(STR_AVERAGE_KEEP_TOKEN_RATIO)
    plt.ylabel(STR_TOP1_ACCURACY)
    plt.legend()
    plt.title(f'{STR_IMAGENET_1K}', fontsize=12)

    plt.savefig(f'{PLOT_NAME}.png', dpi=320)
    plt.savefig(f'{PLOT_NAME}.svg')

    print('done')

if __name__ == '__main__':
    main()