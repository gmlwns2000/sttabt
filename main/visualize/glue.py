def load_samples(subset):
    #batch = {'input_ids', 'attention_mask'}
    pass

def load_models(subset, ltp_config, concrete_config):
    #fix following function to load if already trained.

    #ltp_glue.run_exp_inner(...)
    #concrete_glue.exp_p_logit(...)
    pass

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