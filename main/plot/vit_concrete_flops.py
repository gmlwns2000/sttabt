import os, json, tqdm, torch

from models import sparse_token as sparse
from utils import sparse_flops_calculation as flops_calc
from utils import dyvit_occupy
from matplotlib import pyplot as plt
plt.style.use('seaborn-bright')
from main.plot.constants import *
from main.visualize.vit import load_concrete as load_concrete_model
from trainer import vit_concrete_trainer as vit_concrete

COLOR_DEIT = 'blue'
COLOR_SPVIT = 'magenta'
COLOR_IA = 'brown'
COLOR_S2 = 'black'
COLOR_HVT = 'gray'

def load_points():
    #return [flops] [accuracy] [name] [color]
    ret = [
        #from SPViT paper
        (1.24, 72.2, 'DeiT-T', COLOR_DEIT),
        # (1.0, 72.2, 'SPViT-DeiT-T'),
        # (2.13, 75.00, 'T2T-ViT-10'),
        # (0.93, 72.0, 'PS-ViT'),
        # (0.95, 70.12, 'S$^2$ViTE'),

        (4.58, 79.8, 'DeiT-S', COLOR_DEIT),
        (2.64, 79.34, 'SPViT-DeiT-S', COLOR_SPVIT),
        (3.15, 79.1, 'IA-RED$^2$', COLOR_IA),
        (3.14, 79.22, 'S$^2$ ViTE', COLOR_S2),
        (2.4, 78.00, 'HVT-S-1', COLOR_HVT),
        
        (3.25, 79.0, 'DeiT-S/320', COLOR_DEIT),
        # (2.00, 78.65, 'SPViT-DeiT-S/320'), # this removed since it is not originate from DeiT.
        
        (2.65, 78.53, 'DeiT-S/288', COLOR_DEIT),
        
        (2.14, 77.21, 'DeiT-S/256', COLOR_DEIT),
        # (1.29, 76.87, 'SPViT-DeiT-T/256'),
    ]
    return ([ret[j][i] for j in range(len(ret))] for i in range(len(ret[0])))

def load_dyvit():
    #return [flops] [accuracy]
    ret = []
    for result in dyvit_occupy.DYVIT_RESULTS:
        flops = result['flops']
        acc = result['accuracy']
        ret.append((flops, acc))
    ret = sorted(ret, key=lambda it: it[0])
    return ([ret[j][i] for j in range(len(ret))] for i in range(len(ret[0])))

def load_concrete(factor=4, p_logits=[-3, -2, -1.5, -1.25, -1, -0.5, 0.0, 0.5, 1.0], epochs=20, warmup_epochs=14, device=0):
    #return [flops] [accuracy] [flops_ema] [accuracy_ema]
    ret = []
    dataloader_test = None
    for p in p_logits:
        path_dir = f'./saves/dyvit-concrete-f{factor}-{p}-nohard-e{epochs}-we{warmup_epochs}/'
        path_checkpoint = path_dir + f'checkpoint-{epochs-1}.pth'
        path_log = path_dir + 'log.txt'
        path_flops = path_dir + 'flops-analysis.json'
        if not os.path.exists(path_log):
            print('load_concrete: not found', path_log)
            continue
        
        #load accuracy
        with open(path_log, 'r') as f:
            lines = f.readlines()
            if len(lines) < epochs:
                print(f'load_concrete: train is not finished at epoch {len(lines)+1}')
                continue
            line = lines[-1].strip().strip('\n')
            data = json.loads(line)
            accuracy = data['test_acc1']
            accuracy_ema = data['test_acc1_ema']

        #load flops
        if not os.path.exists(path_flops):
            if not os.path.exists(path_checkpoint):
                print('load_concrete: not found checkpoint', path_checkpoint)
                continue
            
            #calc flops
            #load dataloader_test
            if dataloader_test is None:
                trainer = vit_concrete.VitConcreteTrainer(
                    subset='base', model='deit-small', factor=factor, batch_size=-1, device='cpu',
                    world_size=1, enable_valid=False, epochs=epochs
                )
                dataloader_test = trainer.approx_trainer.timm_data_test
            
            model = load_concrete_model(path_checkpoint, factor=factor, p_logit=p)
            model = model.to(device).eval()

            def evaluate(model):
                sparse.benchmark_reset()
                sparse.benchmark_concrete_occupy(True)
                sparse.benchmark_sparse_approx_flops(True)
                for batch in tqdm.tqdm(dataloader_test, desc=f'p:{p}'):
                    batch = {'pixel_values': batch[0].to(device), 'labels': batch[1].to(device)}
                    batch['output_attentions'] = True
                    batch['output_hidden_states'] = True

                    with torch.no_grad():
                        output = model(**batch)
                flops = sparse.benchmark_get_average('sparse_approx_flops')
                occupy = sparse.benchmark_get_average('concrete_occupy')
                return occupy, flops
            
            occupy, flops = evaluate(model)
            occupy_ema, flops_ema = 0, 0
            data = {
                'occupy': occupy,
                'flops': flops,
                'occupy_ema': occupy_ema,
                'flops_ema': flops_ema
            }
            with open(path_flops, 'w') as f:
                json.dump(data, f, indent=2)
            
            print('load_concrete: json dump', path_flops, data)
        
        with open(path_flops, 'r') as f:
            data = json.load(f)
            flops = data['flops']
            flops_ema = data['flops_ema']
        
        ret.append((flops, accuracy, flops_ema, accuracy_ema))
        
    ret = sorted(ret, key=lambda it: it[0])
    if len(ret) == 0:
        return [], [], [], []
    return ([ret[j][i] for j in range(len(ret))] for i in range(len(ret[0])))

def main():
    xs_dyvit, ys_dyvit = load_dyvit()
    xs_dyvit = scale(xs_dyvit, 1e-9)

    xs_concrete, ys_concrete, xs_concrete_ema, ys_concrete_ema = load_concrete(factor=4)
    xs_concrete_f8, ys_concrete_f8, xs_concrete_ema_f8, ys_concrete_ema_f8 = load_concrete(factor=8)

    xs_other, ys_other, labels_other, colors_other = load_points()

    plt.clf()
    fig, ax = plt.subplots()
    plt.plot(
        xs_dyvit, ys_dyvit, 
        label=STR_DYNAMIC_VIT, color='orange',
        marker='o', linestyle='-', linewidth=1.2, zorder=1,
    )
    plt.plot(
        xs_concrete, ys_concrete, 
        label="STTABT@f4 (Concrete) DeiT-S", color=COLOR_STTABT_CONCRETE_WITH_TRAIN,
        marker='^', linestyle='--', linewidth=1.2, zorder=99,
    )
    if len(xs_concrete_f8) > 0:
        plt.plot(
            xs_concrete_f8, ys_concrete_f8, 
            label="STTABT@f8 (Concrete) DeiT-S", color=COLOR_STTABT_CONCRETE_WITH_TRAIN,
            marker='^', linestyle='-', linewidth=1.2, zorder=99,
        )
    
    for i, txt in enumerate(labels_other):
        plt.scatter(xs_other[i], ys_other[i], color=colors_other[i])
        ax.annotate(txt, (xs_other[i], ys_other[i]), fontsize=6)
    
    plt.legend(prop={'size': 9})
    plt.title(f'{STR_IMAGENET_1K}', fontsize=12)
    plt.xlabel('GFLOPs')
    plt.ylabel(STR_TOP1_ACCURACY)

    y_bot, y_top = plt.ylim()
    y_bot = y_top - (y_top-y_bot) * 0.96
    plt.ylim(y_bot, y_top)
    
    plt.grid(which='both', axis='both')

    filename = './saves_plot/vit-flops'
    plt.savefig(filename+'.png')
    plt.savefig(filename+'.pdf')

if __name__ == '__main__':
    main()