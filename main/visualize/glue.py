from main import ltp_glue_plot, concrete_glue_plot

def load_samples(subset):
    #batch = {'input_ids', 'attention_mask'}
    pass

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

def vis_glue(subset='sst2'):
    configs = {
        'sst2': {
            #use ltp.sst2[2]
            #use concrete.sst2[1]
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

    model_ltp, model_concrete = load_models(subset, ltp_config, concrete_config)
    lines, batch = load_samples(subset)

    #run, gather masks

    #visualize

def main():
    vis_glue('sst2')

if __name__ == '__main__':
    main()