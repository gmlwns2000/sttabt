import argparse
import subprocess
import sys,os,signal
import time
from datetime import datetime

from utils import env_vars

__print = print
__print_fd = None
__print_log_dir = './saves/vit_concrete_end2end_logs'
if not os.path.exists(__print_log_dir):
    os.mkdir(__print_log_dir)
__print_log_path = f'{__print_log_dir}/{datetime.now().strftime("%Y-%d-%m-%H-%M-%S-%f")}.log'

def print(*args, **kwargs):
    global __print, __print_fd, __print_log_dir, __print_log_path
    __print(*args, **kwargs)
    try:
        if __print_fd is None:
            __print_fd = open(__print_log_path, 'w', buffering=1)
        __print_fd.write(" ".join([str(a) for a in args]))
        __print_fd.write(kwargs.get('end', '\n'))
        __print_fd.flush()
    except Exception as ex:
        __print('error while print out to log', ex)

def log(*args):
    print('VitConcreteEnd2End:', *args)

def execute_proc(cmd, new_envs=None):
    log('execute', cmd)
    #return_code = subprocess.call(cmd, shell=True)
    #return 0
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = my_env.get("PYTHONPATH", '') + ":./"
    if new_envs is not None:
        for key in new_envs:
            my_env[key] = new_envs[key]
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
    assert args.model in ['lvvit-small', 'deit-small']
    
    cmd = f"python -m trainer.vit_approx_trainer --factor {args.factor} --n-gpus {args.n_gpus} --epochs {args.approx_epochs} --model {args.model}"
    return_code = execute_proc(cmd, new_envs={
        'IMAGENET_ROOT': args.imagenet_root
    })
    log('Approx Train Finished!')
    if return_code != 0:
        raise Exception('approx trainer failed to run with return code', return_code)

def run_p_logits(args):
    p_logits = [-3, -2, -1.5, -1.25, -1, -0.5, 0.0, 0.5, 1.0]
    if not args.p_logits is None:
        p_logits = [float(i) for i in args.p_logits.split(' ')]
    log('P_logits to train:', p_logits)

    for i, p in enumerate(p_logits):
        t = time.time()
        log('current p_logit,', p)
        update_freq = args.concrete_total_batch_size // (args.batch_size * args.n_gpus)
        if update_freq <= 0:
            log('WARN: update freq seems less then 1. Did you put too much batch_size? update_freq update to 1.')
            update_freq = 1
        
        output_dir = f"./saves/dyvit-concrete-f{args.factor}-{p}-nohard-e{args.epochs}-we{args.warmup_epochs}"
        if args.model != 'deit-small':
            output_dir = f"./saves/dyvit-concrete-{args.model}-f{args.factor}-{p}-nohard-e{args.epochs}-we{args.warmup_epochs}"
        cmd = f"python -m torch.distributed.launch --nproc_per_node={args.n_gpus} --master_port={args.master_port} --use_env trainer/dyvit_concrete_trainer.py --batch_size {args.batch_size} --update_freq {update_freq} --warmup_epochs {args.warmup_epochs} --epochs {args.epochs} --p-logit {p} --max-hard-train-epochs 0 --approx-factor {args.factor} --auto_resume {str(args.auto_resume).lower()} --output_dir {output_dir} --data_path=\"{args.imagenet_root}\" --model {args.model}"
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
    parser.add_argument('--approx-epochs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-epochs', type=int, default=14)
    parser.add_argument('--n-gpus', type=int, default=8)
    parser.add_argument('--imagenet-root', type=str, default=env_vars.get_imagenet_root())
    parser.add_argument('--model', type=str, default='deit-small')
    parser.add_argument('--master-port', type=int, default=12127)
    parser.add_argument('--skip-approx', action='store_true', default=False)
    parser.add_argument('--auto-resume', action='store_true', default=False)
    parser.add_argument('--concrete-total-batch-size', type=int, default=512)
    parser.add_argument('--p-logits', type=str, default=None, help="ex: \"-1.0 0.0 1.0\"")

    args = parser.parse_args()
    log(args)

    if not args.skip_approx:
        run_approx(args)
    else:
        log('approx train is skipped')
    
    run_p_logits(args)

if __name__ == '__main__':
    main()