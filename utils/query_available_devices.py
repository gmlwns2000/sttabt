import multiprocessing as mp

def __query_available_devices(q):
    import torch
    num_gpus = torch.cuda.device_count()
    available_devices = []
    avail_mem = []
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        if (free_mem / total_mem) > 0.8:
            available_devices.append(i)
        avail_mem.append(f'[{i}]={free_mem / total_mem * 100:.1f}%')
    print('QueryDevice: Available Memories,', *avail_mem)
    q.put(available_devices)

def query_available_devices():
    q = mp.Queue()
    cuda_process = mp.Process(target=__query_available_devices, args=(q,), daemon=True)
    cuda_process.start()
    cuda_process.join()
    available_devices = q.get()
    q.close()
    return available_devices