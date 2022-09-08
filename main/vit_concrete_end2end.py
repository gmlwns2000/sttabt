import argparse
import subprocess
import sys,os,signal
import time

def log(*args):
    print('VitConcreteEnd2End:', *args)

def execute_proc(cmd):
    log('execute', cmd)
    #return_code = subprocess.call(cmd, shell=True)
    #return 0
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = my_env.get("PYTHONPATH", '') + ":./"
    with subprocess.Popen(cmd, 
        stdout=subprocess.PIPE, 
        universal_newlines=True, 
        shell=True, 
        preexec_fn=os.setsid, 
        #stderr=subprocess.STDOUT,
        env=my_env,
    ) as popen:
        try:
            exited = False
            for stdout_line in iter(popen.stdout.readline, ""):
                print(stdout_line, end='')
                if 'EXIT_PATTERN' in stdout_line:
                    exited = True
                    os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
            popen.stdout.close()
            return_code = popen.wait()
            if return_code and not exited:
                return return_code
            return 0
        finally:
            try:
                os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
            except ProcessLookupError as ex:
                log('process exited')

def run_approx(args):
    cmd = f"python -m trainer.vit_approx_trainer --factor {args.factor} --n-gpus {args.n_gpus}"
    return_code = execute_proc(cmd)
    if return_code != 0:
        raise Exception('approx trainer failed to run with return code', return_code)

def run_p_logits(args):
    p_logits = [-3, -2, -1.5, -1.25, -1, -0.5, 0.0, 0.5, 1.0]
    log('P_logits to train:', p_logits)

    for i, p in enumerate(p_logits):
        t = time.time()
        log('current p_logit,', p)
        update_freq = args.concrete_total_batch_size // (args.batch_size * args.n_gpus)
        if update_freq <= 0:
            log('WARN: update freq seems less then 1. Did you put too much batch_size? update_freq update to 1.')
            update_freq = 1
        
        cmd = f"python -m torch.distributed.launch --nproc_per_node={args.n_gpus} --master_port={args.master_port} --use_env trainer/dyvit_concrete_trainer.py --batch_size {args.batch_size} --update_freq {update_freq} --warmup_epochs {args.warmup_epochs} --epochs {args.epochs} --p-logit {p} --max-hard-train-epochs 0 --approx-factor {args.factor} --auto_resume {str(args.auto_resume).lower()} --output_dir ./saves/dyvit-concrete-f{args.factor}-{p}-nohard-e{args.epochs}-we{args.warmup_epochs}"
        return_code = execute_proc(cmd)
        if return_code != 0:
            raise Exception('concrete trainer failed to run with return code', return_code)
        elapsed = time.time() - t
        log('Train took', elapsed, 'sec')
        log('ETA:', elapsed*(len(p_logits)-i-1)/3600, 'hrs')
    log('finished')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-epochs', type=int, default=14)
    parser.add_argument('--n-gpus', type=int, default=8)
    parser.add_argument('--master-port', type=int, default=12127)
    parser.add_argument('--skip-approx', action='store_true', default=False)
    parser.add_argument('--auto-resume', action='store_true', default=False)
    parser.add_argument('--concrete-total-batch-size', type=int, default=512)

    args = parser.parse_args()
    log(args)

    if not args.skip_approx:
        run_approx(args)
    else:
        log('approx train is skipped')
    
    run_p_logits(args)

if __name__ == '__main__':
    main()