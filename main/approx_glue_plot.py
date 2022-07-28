"""
This script automatically assign all gpus and exam.
기기의 메모리가 남는 모든 gpu를 자동으로 할당합니다.
"""

import time
import traceback
import trainer.glue_base as glue_base
import models.sparse_token as sparse
from utils.glue import get_score
import pickle, importlib, itertools, gc, json, random
import torch
from matplotlib import pyplot as plt
import multiprocessing as mp
import copy
plt.style.use("seaborn")
sparse.set_update_input_mask_accumulate_indices(True)

Glue = glue_base.GlueAttentionApproxTrainer

plot_header = '[F8-PREWIKI.v2]'
result_name = None
result_pkl = None
def update_result_names():
    global plot_header, result_name, result_pkl
    result_name = 'saves_plot/' + plot_header + 'glue_benchmark_accum_absatt'
    result_pkl = result_name + '.pickle'
update_result_names()

factor = 8
subsets = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli",]
kss = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75]

#subsets = ['cola', 'mrpc', 'mrpc', 'mrpc', 'mrpc', 'mrpc', 'mrpc', 'mrpc']
#kss = [0.1, 0.5]

MAX_STEP = 50

def merge_dict(a, b):
    a = copy.deepcopy(a)
    a.update(b)
    return a

def run_exp_subset(ret_queue, iset, subset, kss, cases_len, run_approx, device):
    gc.collect()
    torch.cuda.empty_cache()
    
    results = {}

    trainer = Glue(subset, factor=factor, batch_size=1, wiki_train=False, device=device)
    if run_approx: trainer.load()
    bert_score, _ = get_score(trainer.eval_base_model(max_step=MAX_STEP))
    results[(subset, 'bert')] = { 'score_bert':bert_score }
    print('bert', bert_score)
    
    for ik, ks in enumerate(kss):
        i = iset * len(kss) + ik
        
        trainer.set_batch_size(1)
        ksx = [ks for _ in range(12)]
        sparse.benchmark_reset()
        score_sparse, metric = get_score(trainer.eval_sparse_model(ks=ksx, run_original_attention=True, show_message=False, max_step=MAX_STEP))
        mask_occupy = sparse.benchmark_get_average('mask_occupy')
        flops_sparse = sparse.benchmark_get_average('sparse_approx_flops')
        print('sparse absatt', score_sparse, '@', mask_occupy)

        if run_approx:
            trainer.set_batch_size(1)
            ksx = [ks for _ in range(12)]
            sparse.benchmark_reset()
            score_approx, metric = get_score(trainer.eval_sparse_model(ks=ksx, run_original_attention=False, show_message=False, max_step=MAX_STEP))
            mask_occupy_approx = sparse.benchmark_get_average('mask_occupy')
            flops_approx = sparse.benchmark_get_average('sparse_approx_flops')
            print('sparse approx', score_approx, '@', mask_occupy_approx)

        trainer.set_batch_size(8)
        target_ks = mask_occupy
        if target_ks <= 0.666:
            ksx = [target_ks*0.5+((1-x/10.0)**1.0) * target_ks for x in range(12)]
        else:
            ksx = [(1-x/10.0)*(2-2*target_ks)+(2*target_ks-1) for x in range(12)]
        #ksx[-1] = 0.99
        sparse.benchmark_reset()
        score_forward, _ = get_score(trainer.eval_sparse_model(ks=ksx, use_forward=True, show_message=False, max_step=MAX_STEP))
        mask_occupy_forward = sparse.benchmark_get_average('forward_occupy')
        flops_forward = sparse.benchmark_get_average('sparse_approx_flops')
        print('forward', score_forward, '@', mask_occupy_forward)

        result = {
            'occupy':mask_occupy, 'score_sparse':score_sparse, 'flops_sparse': flops_sparse,
            'occupy_forward': mask_occupy_forward, 'score_forward':score_forward, 'flops_forward': flops_forward,
            'metric':metric
        }
        if run_approx:
            result['score_sparse_approx'] = score_approx
            result['occupy_approx'] = mask_occupy_approx
            result['flops_approx'] = flops_approx
        print(f"\nRESULT {subset}@{ks} ({i+1}/{cases_len}) |", result)
        results[(subset, ks)] = result
    
    ret_queue.put(results)

def runtime_wrapper(ret_queue, fn, *args):
    try:
        fn(ret_queue, *args)
    except Exception as ex:
        traceback.print_exc()
        print('exception occure with', args)
        ret_queue.put({
            'status': 'failed',
            'ex': ex,
            'args': args,
        })

def run_exp(run_approx=True, devices=[0], subsets=subsets, retry=5):
    if retry < 0:
        raise Exception('Exhausted retry')
    assert len(devices) > 0

    global kss

    results = {}
    cases = list(itertools.product(subsets, kss))
    available_devices = set(devices)
    running_devices = set()
    procs = []
    ret_queue = mp.Queue(maxsize=1024)
    def check_procs():
        for proc in procs:
            p = proc['p'] # type: mp.Process
            dev = proc['device']
            if not p.is_alive():
                running_devices.remove(dev)
                available_devices.add(dev)
                procs.remove(proc)
                break
    
    for iset, subset in enumerate(subsets):
        while len(available_devices) < 1:
            #check for end
            check_procs()
            time.sleep(0.01)
        
        target_device = random.sample(available_devices, 1)[0]
        available_devices.remove(target_device)
        running_devices.add(target_device)

        proc = mp.Process(
            target=runtime_wrapper, 
            args=(ret_queue, run_exp_subset, iset, subset, kss, len(cases), run_approx, target_device),
            daemon=True
        )
        proc.start()
        procs.append({'p': proc, 'device':target_device})
    
    while len(procs) > 0:
        check_procs()
        time.sleep(0.01)
    
    retry_subsets = []
    while not ret_queue.empty():
        result = ret_queue.get()
        if isinstance(result, dict):
            if result['status'] == 'failed':
                print(f'Process failed with following argument, {result["args"]}')
                print(f"Process exception: {result['ex']}")
                subset = result['args'][3]
                print(f'Retry subset {subset}')
                retry_subsets.append(subset)
        else:
            for k, v in result.items():
                results[k] = v
    
    if len(retry_subsets) > 0:
        print('Pending retries...', retry_subsets)
        time.sleep(5)
        retry_results = run_exp(run_approx=run_approx, devices=devices, subsets=retry_subsets, retry=retry-1)
        results.update(retry_results)

    with open(result_pkl, 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_from_last_pickle():
    with open(result_pkl, 'rb') as f:
        results = pickle.load(f)

    run_approx = 'occupy_approx' in results[list(results.keys())[-1]]

    for subset in subsets:
        ys_sparse = []
        ys_approx = []
        ys_forward = []
        xs_sparse = []
        xs_approx = []
        xs_forward = []
        xs_sparse_flops = []
        xs_approx_flops = []
        xs_forward_flops = []
        metric_name = None
        for ks in kss:
            item = results[(subset, ks)]
            metric_name = item['metric']
            
            xs_sparse.append(item['occupy'])
            xs_sparse_flops.append(item['flops_sparse'])
            ys_sparse.append(item['score_sparse'])
            
            xs_forward.append(item['occupy_forward'])
            xs_forward_flops.append(item['flops_forward'])
            ys_forward.append(item['score_forward'])

            if run_approx: 
                xs_approx.append(item['occupy_approx'])
                xs_approx_flops.append(item['flops_approx'])
                ys_approx.append(item['score_sparse_approx'])

        xs_bert = [min(xs_sparse+xs_approx+xs_forward), max(xs_sparse+xs_approx+xs_forward)]
        xs_bert_flops = [min(xs_sparse_flops+xs_approx_flops+xs_forward_flops), max(xs_sparse_flops+xs_approx_flops+xs_forward_flops)]
        ys_bert = [results[(subset, 'bert')]['score_bert'],]*2

        plt.clf()
        plt.plot(xs_sparse, ys_sparse, marker='o', label='sparse (abs.att.)')
        if run_approx: 
            plt.plot(xs_approx, ys_approx, marker='o', label='sparse (approx.)')
        plt.plot(xs_forward, ys_forward, marker='o', label='forward only')
        plt.plot(xs_bert, ys_bert, linestyle='--', label='bert-base')
        plt.xlabel('occupy')
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(f'{subset}')
        plt.savefig(f'{result_name}_{subset}.png', dpi=320)

        plt.clf()
        plt.plot(xs_sparse_flops, ys_sparse, marker='o', label='sparse (abs.att.)')
        if run_approx: 
            plt.plot(xs_approx_flops, ys_approx, marker='o', label='sparse (approx.)')
        plt.plot(xs_forward_flops, ys_forward, marker='o', label='forward only')
        plt.plot(xs_bert_flops, ys_bert, linestyle='--', label='bert-base')
        plt.xlabel('GFLOPs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(f'{subset}')
        plt.savefig(f'{result_name}_flops_{subset}.png', dpi=320)

def query_available_devices(q):
    import torch
    num_gpus = torch.cuda.device_count()
    available_devices = []
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        if (free_mem / total_mem) > 0.97:
            available_devices.append(i)
    q.put(available_devices)

def main():
    global plot_header, factor

    #arg
    import argparse, random

    parser = argparse.ArgumentParser()
    parser.add_argument('--header', type=str, default=None)
    parser.add_argument('--factor', type=int, default=None)

    args = parser.parse_args()
    if args.header is not None:
        plot_header = args.header
        update_result_names()
    if args.factor is not None:
        factor = args.factor
    
    q = mp.Queue()
    cuda_process = mp.Process(target=query_available_devices, args=(q,), daemon=True)
    cuda_process.start()
    cuda_process.join()
    available_devices = q.get()
    q.close()
    print('Available Devices', available_devices)

    run_exp(devices=available_devices)
    plot_from_last_pickle()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()