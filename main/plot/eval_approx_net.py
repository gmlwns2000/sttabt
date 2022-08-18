import torch, os, random, time, json, tqdm
import multiprocessing as mp
import numpy as np
import pandas as pd

from main.plot.constants import GLUE_SUBSETS, SUBSET_TO_NAME
from trainer import glue_base as glue
from utils.gpu_pool import GPUPool
from utils.load_state_dict_interpolated import load_state_dict_interpolated

EPS=1e-7

def eval_approx_net_subset(device, tqdm_position, factor, trained, subset):
    trainer = glue.GlueAttentionApproxTrainer(
        dataset=subset, 
        factor=factor, 
        device=device,
        wiki_train=False,
    )
    if trained == 'load':
        trainer.load()
    elif trained == 'interpolate':
        load_state_dict_interpolated(trainer.approx_bert.module, trainer.model.state_dict(), ignores=['p_logit', 'ltp', 'transfer_'])
    else:
        pass

    mse_sum = 0
    kl_sum = 0
    loss_count = 0
    
    trainer.approx_bert.eval()
    trainer.approx_bert = trainer.approx_bert.module
    trainer.model.eval()
    for step, batch in enumerate(tqdm.tqdm(trainer.test_dataloader, position=tqdm_position, desc=f"Eval[{subset}]")):
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
            batch['output_attentions'] = True
            if 'labels' in batch: del batch['labels']
            target_attentions = trainer.model(**batch).attentions

            student_attentions = trainer.approx_bert(**batch).attentions
            for j in range(len(target_attentions)):
                N, H, T, T = student_attentions[j].shape

                attention_mask = batch['attention_mask']
                assert attention_mask.shape == (N, T)

                mse = torch.square(student_attentions[j]- target_attentions[j])
                mse = torch.sum(mse * attention_mask.view(N, 1, 1, T), dim=-1) / torch.sum(attention_mask, dim=-1).view(N, 1, 1)
                mse = torch.sum(mse * attention_mask.view(N, 1, T), dim=-1) / torch.sum(attention_mask, dim=-1).view(N, 1)
                mse = mse.mean()
                mse_sum += mse.item()
                
                y_pred = student_attentions[j].view(N*H*T, T)
                y_target = target_attentions[j].view(N*H*T, T)
                kl_loss = y_target * ((y_target + EPS).log() - (y_pred + EPS).log())
                kl_loss = torch.sum(kl_loss.view(N, H, T, T), dim=-1) # shape: N, H, T
                
                kl_loss = kl_loss * attention_mask.view(-1, 1, T)
                kl_loss = torch.sum(kl_loss, dim=-1)
                kl_loss = kl_loss / torch.sum(attention_mask, dim=-1).view(N, 1)
                kl_loss = kl_loss.mean() # head and batch mean
                kl_sum += kl_loss.item()

                loss_count += 1
    
    mse = mse_sum / loss_count
    kl = kl_sum / loss_count
    print(f'EvalApproxSubset[{subset}]: KLDiv: {kl}, MSE: {mse}')

    return {
        'subset': subset,
        'trained': trained,
        'factor': factor,
        'kldiv': kl,
        'mse': mse,
    }

def eval_approx_net(factor, trained):
    args = GLUE_SUBSETS
    args = [(factor, trained, a) for a in args]

    pool = GPUPool(name=f'Factor{factor}:{trained}')
    results = pool.run(eval_approx_net_subset, args)

    ret = {}
    for result in results:
        ret[result['subset']] = {
            'kldiv': result['kldiv'],
            'mse': result['mse'],
        }
    return ret

def exam():
    cases = [
        (2, 'random'),
        (4, 'random'),
        (2, 'interpolate'),
        (4, 'interpolate'),
        (4, 'load'),
        (8, 'load'),
    ]
    ret = {}
    for case in cases:
        result = eval_approx_net(case[0], case[1])
        t = ret.get(case[1], {})
        t[case[0]] = result
        ret[case[1]] = t
    
    with open(f'saves_plot/approx-attention-glue-evaluation.json', 'w') as f:
        json.dump(ret, f, indent=2)

def plot():
    with open(f'saves_plot/approx-attention-glue-evaluation.json', 'r') as f:
        data = json.load(f)
    
    init_mode_to_name = {
        'random': 'Random',
        'interpolate': 'Interpolate',
        'load': 'Distilled',
    }
    rows = []
    for init_mode in data.keys():
        for factor in data[init_mode].keys():
            items = data[init_mode][factor]
            row = []
            for subset in GLUE_SUBSETS:
                item = items.get(subset, {'mse':0.0, 'kldiv':0.0})
                row.append(item['mse'])
                row.append(item['kldiv'])
            rows.append((
                (init_mode_to_name[init_mode], factor), row
            ))
    
    df = pd.DataFrame()
    for header, content in rows:
        df[header] = content
    df.columns = pd.MultiIndex.from_tuples([(c[0], c[1]) for c in df.columns])
    inds = []
    for subset in GLUE_SUBSETS:
        inds.append((SUBSET_TO_NAME[subset], 'MSE'))
        inds.append((SUBSET_TO_NAME[subset], 'KL Div.'))
    df.index = pd.MultiIndex.from_tuples(inds)
    #df = df.transpose()
    styler = df.style
    styler.applymap_index(lambda v: "font-weight: bold;", axis="index")
    styler.applymap_index(lambda v: "font-weight: bold;", axis="columns")
    styler.format(na_rep='MISS', precision=4)
    def highlight_min(s, props=''):
        return np.where(s == np.nanmin(s.values), props, '')
    styler.apply(highlight_min, props='font-weight: bold;', axis=1)
    styler.to_latex('./saves_plot/table_approx_att_eval.tex', convert_css=True)
    #df.to_latex('./saves_plot/table_approx_att_eval.tex', float_format='%.4f')
    df.plot()
    from matplotlib import pyplot as plt
    plt.show()
    print(df)

def main():
    #exam()
    plot()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()