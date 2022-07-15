import torch, time, gc

# Timing utilities
timer_dict = {}

def start(name='default'):
    global timer_dict
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()
    start_mem = torch.cuda.max_memory_allocated()
    timer_dict[name] = {
        'time': start_time,
        'mem':start_mem,
    }

def end(name='default'):
    global timer_dict
    torch.cuda.synchronize()
    end_time = time.time()
    end_mem = torch.cuda.max_memory_allocated()

    elapsed = end_time - timer_dict[name]['time']
    memory_allocated = end_mem - timer_dict[name]['mem']

    return elapsed, memory_allocated