import os, gc, torch, cv2, io
import transformers
import numpy as np
from matplotlib import pyplot as plt

from models import sparse_token as sparse
from main import ltp_glue_plot, concrete_glue_plot

def load_samples(subset, tokenizer: "transformers.BertTokenizerFast"):
    #batch = {'input_ids', 'attention_mask'}
    txt_path = f'./samples/glue/{subset}.txt'
    assert os.path.exists(txt_path)
    with open(txt_path, 'r') as f:
        lines = f
        lines = [line.strip().strip('\n').strip('"') for line in lines]
    output = tokenizer(lines, return_tensors='pt', padding=True)
    return lines, {'input_ids': output.input_ids, 'attention_mask': output.attention_mask}

def load_models(subset, ltp_config, concrete_config, device=0, tqdm_position=0, factor=4):
    #ltp_glue.run_exp_inner(...)
    _, ltp_trainer = ltp_glue_plot.run_exp_inner(
        device=device, tqdm_position=tqdm_position, subset=subset, batch_size=-1,
        ltp_lambda=ltp_config['lambda'], ltp_temperature=ltp_config['temperature'], 
        restore_checkpoint=True, return_trainer=True, skip_eval=True
    )

    #concrete_glue.exp_p_logit(...)
    _, concrete_trainer = concrete_glue_plot.exp_p_logit(
        device=device, tqdm_position=tqdm_position, i=0, 
        subset=subset, factor=factor, batch_size=-1, 
        p_logit=concrete_config['p_logit'], lr_multiplier=concrete_config['lr_multiplier'], 
        epochs_multiplier=concrete_config['epochs_multiplier'], grad_acc_multiplier=concrete_config['grad_acc_multiplier'], 
        eval_valid=False, eval_test=False, restore_checkpoint=True, return_trainer=True
    )

    return ltp_trainer.tokenizer, ltp_trainer.sparse_bert.module.to('cpu'), concrete_trainer.sparse_bert.module.to('cpu')

@torch.no_grad()
def mask_concrete(batch, model: "sparse.SparseBertForSequenceClassification"):
    N, TLEN = batch['input_ids'].shape
    model = model.eval()
    model.bert.set_concrete_hard_threshold(0.5)

    output = model(**batch)

    masks = []
    for layer in model.bert.encoder.layer:
        layer = layer #type: sparse.BertLayer
        mask = layer.attention.get_attention().concrete_input_mask.view(N, TLEN)
        masks.append(mask)
    masks = torch.stack(masks, dim=1)

    # for ib in range(N):
    #     print(ib)
    #     print("\n".join(str(" ".join([str(int(masks[ib][i, j].item())) for j in range(TLEN)])) for i in range(12)))

    return masks

@torch.no_grad()
def mask_ltp(batch, model: "sparse.SparseBertForSequenceClassification"):
    N, TLEN = batch['input_ids'].shape
    model = model.eval()
    model.bert.set_ltp_prune_token_soft_pruning(False)

    output = model(**batch)

    masks = [torch.ones((N, TLEN), dtype=torch.float32)]
    for i, layer in enumerate(model.bert.encoder.layer):
        layer = layer #type: sparse.BertLayer
        if i < (len(model.bert.encoder.layer) - 1):
            mask = layer.ltp_prune_token_module.last_mask.view(N, TLEN) #this is output mask
            masks.append(mask)
    masks = torch.stack(masks, dim=1)

    # for ib in range(N):
    #     print(ib)
    #     print("\n".join(str(" ".join([str(int(masks[ib][i, j].item())) for j in range(TLEN)])) for i in range(12)))

    return masks

def render_mask(tokenizer: "transformers.BertTokenizerFast", ids, attention_mask, mask, title, filename):
    TLEN = int(np.sum(attention_mask))
    NLAYER, _ = mask.shape

    labels_vertical = [f"Layer {i+1}" for i in range(NLAYER)]
    labels_horizontal = [tokenizer.decode(ids[i]) for i in range(TLEN)]
    mask = mask[:, :len(labels_horizontal)]

    plt.clf()

    fig, ax = plt.subplots()
    ax.imshow(mask)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels_horizontal)))
    ax.set_yticks(np.arange(len(labels_vertical)))
    ax.set_xticklabels(labels_horizontal)
    ax.set_yticklabels(labels_vertical)

    # Rotate the tick labels and set their alignment.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right", rotation_mode="anchor")

    ax.set_title(title)
    fig.tight_layout()
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.savefig(filename + '.png', bbox_inches='tight',pad_inches=0.1)
    plt.savefig(filename + '.pdf', bbox_inches='tight',pad_inches=0.1)
    print('render_mask:', filename, '.pdf, .png')
    
    return img

def render_masks(tokenizer, batch, masks, title, filenames):
    plots = []
    for i in range(batch['input_ids'].shape[0]):
        plots.append(render_mask(
                tokenizer, 
                batch['input_ids'][i].cpu().numpy(), 
                batch['attention_mask'][i].cpu().numpy(), 
                masks[i].cpu().numpy(), 
                title,
                filenames[i]
        ))
    return plots

def vis_glue(subset='sst2'):
    configs = {
        'sst2': {
            #use ltp.sst2[2] occ: 20%
            #use concrete.sst2[1] occ: 12%
            #similar accuracy about 89%
            'ltp': {
                "lambda": 0.1,
                "temperature": 0.002,
            },
            'concrete': {
                "p_logit": -1.5,
                "lr_multiplier": 1.0,
                "epochs_multiplier": 1.0,
                "grad_acc_multiplier": 1.0,
            }
        }
    }
    ltp_config = configs[subset]['ltp']
    concrete_config = configs[subset]['concrete']

    tokenizer, model_ltp, model_concrete = load_models(subset, ltp_config, concrete_config)
    lines, batch = load_samples(subset, tokenizer)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    #run, gather masks
    masks_concrete = mask_concrete(batch, model_concrete)
    masks_ltp = mask_ltp(batch, model_ltp)

    #visualize
    plots_concrete = render_masks(
        tokenizer, batch, masks_concrete, "STTABT (Concrete)",
        [f'./saves_plot/visualization_nlp/{i}_concrete' for i in range(len(masks_concrete))]
    )
    plots_ltp = render_masks(
        tokenizer, batch, masks_ltp, "LTP",
        [f'./saves_plot/visualization_nlp/{i}_ltp' for i in range(len(masks_concrete))]
    )

def main():
    vis_glue('sst2')

if __name__ == '__main__':
    main()