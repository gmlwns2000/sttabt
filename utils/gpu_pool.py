import tqdm, time, math, random
import multiprocessing as mp
import traceback
from utils import query_available_devices
from collections.abc import Iterable

def dummy_fn(device, tqdm_position, *args):
    """Pool Inner Function. 
    The experiment function should follows this format.

    Args:
        device (int): device id
        tqdm_position (int): tqdm position.
            Use like this `tqdm.tqdm(position=tqdm_poisiton)`

    Returns:
        Any: the return value of experiment. This should be pickleable.
    """

    for _ in tqdm.tqdm(range(1000), desc=f'dev:{device}, args:{args}', position=tqdm_position):
        time.sleep(1/500)
    time.sleep(random.random() * 3)
    
    if random.random() < 0.1:
        raise Exception('random exception')
    
    return random.randint(0, 1000)

def print(*args):
    args = [a if isinstance(a, str) else str(a) for a in args]
    tqdm.tqdm.write(" ".join(args))

def runtime_wrapper(ret_queue: "mp.Queue", tqdm_lock: "mp.RLock", fn, device, tqdm_position, *args):
    tqdm.tqdm.set_lock(tqdm_lock)
    args_text = ''
    try:
        args_text = str(args)
        if len(args_text) > 50:
            args_text = args_text[:47] + '...'
        print(f'Runtime[{device}]: Started with args{args_text}')
        ret = fn(device, tqdm_position, *args)
        ret_queue.put(ret, timeout=5)
        print(f'Runtime[{device}]: Finished with args{args_text}')
    except Exception as ex:
        print(f'Runtime[{device}]: Failed with following exception. {ex}')
        ret_queue.put({
            'status': 'failed',
            'ex': ex,
            'args': args,
            'args_string': args_text,
            'device': device,
            'tqdm_position': tqdm_position,
            'traceback': traceback.format_exc()
        })

class GPUPool:
    def __init__(self, devices=None):
        if devices is None:
            devices = query_available_devices()
        self.devices = devices
        self.retries = 3
        self.queue = None

        if not 'RLock' in str(tqdm.tqdm.get_lock()):
            print('GPUPool: tqdm write lock is not mp.RLock. Initialized.')
            tqdm.tqdm.set_lock(mp.RLock())
    
    def run(self, fn, args_list: list):
        if self.queue is not None:
            self.queue.close()
        args_list = list([args if isinstance(args, Iterable) else [args] for args in args_list])
        self.queue = mp.Queue()

        return self.__run(fn=fn, args_list=args_list, retry=self.retries)

    def __run(self, fn, args_list, retry):
        if retry < 0:
            raise Exception('Retry failed')
        
        procs = [] #type: list[mp.Process]
        try:
            results = []
            available_devices = set(self.devices)
            running_devices = set()
            def check_procs():
                for proc in procs:
                    p = proc['p'] # type: mp.Process
                    dev = proc['device']
                    if not p.is_alive():
                        running_devices.remove(dev)
                        available_devices.add(dev)
                        procs.remove(proc)
                        break
            
            for args in args_list:
                while len(available_devices) < 1:
                    check_procs()
                    time.sleep(1/1000)
                
                target_device = random.sample(available_devices, 1)[0]
                available_devices.remove(target_device)
                running_devices.add(target_device)
                
                proc = mp.Process(
                    target=runtime_wrapper, 
                    args=(
                        self.queue, tqdm.tqdm.get_lock(), fn, target_device, 
                        self.devices.index(target_device), *args ),
                    daemon=True
                )
                proc.start()
                procs.append({'p': proc, 'device':target_device})
            
            while len(procs) > 0:
                check_procs()
                time.sleep(1/1000)
            
            retry_args_list = []
            while not self.queue.empty():
                result = self.queue.get()
                if isinstance(result, dict) and 'status' in result and result['status'] == 'failed':
                    print(f"GPUPool: Process failed on device {result['device']} with following argument, {result['args_string']}")
                    print(f"GPUPool: Process traceback: {result['traceback']}")
                    print(f"GPUPool: Process exception: {result['ex']}")
                    retry_args_list.append(result['args'])
                else:
                    results.append(result)
            
            assert self.queue.empty()
            
            if len(retry_args_list) > 0:
                print(f'Pending retry args {len(retry_args_list)}, left chance {retry}. Wait for 5 seconds')
                time.sleep(5)
                retry_results = self.__run(
                    fn=fn, args_list=retry_args_list, retry=retry-1
                )
                results += retry_results

            return results
        finally:
            for proc in procs:
                proc.kill()

if __name__ == '__main__':
    pool = GPUPool(devices=list(range(4)))
    ret = pool.run(dummy_fn, list(range(18)))
    assert len(ret) == 18
    print('Pool result:', ret)
    ret = pool.run(dummy_fn, list(range(18)))
    assert len(ret) == 18
    print('Pool result:', ret)