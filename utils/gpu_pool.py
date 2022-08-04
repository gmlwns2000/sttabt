import tqdm, time, math, random, sys, threading
import multiprocessing as mp
import traceback
from utils import query_available_devices
from collections.abc import Iterable
from contextlib import contextmanager

__write_lock = None #type: mp.RLock
__clear_event = None #type: mp.Event
__refresh_event = None #type: mp.Event
__clear_thread = None #type: threading.Thread
__refresh_thread = None #type: threading.Thread
def initialize(write_lock=None, clear_event=None, refresh_event=None):
    global __write_lock, __clear_event, __refresh_event, __clear_thread, __refresh_thread
    if write_lock is None:
        write_lock = mp.RLock()
    if clear_event is None:
        clear_event = mp.Event()
    if refresh_event is None:
        refresh_event = mp.Event()
    __write_lock = write_lock
    __clear_event = clear_event
    __refresh_event = refresh_event

    __clear_thread = threading.Thread(target=__thread_clear_event, daemon=True)
    __clear_thread.start()
    __refresh_thread = threading.Thread(target=__thread_refresh_event, daemon=True)
    __refresh_thread.start()

def is_initialized():
    global __write_lock
    return __write_lock is not None

__cleared_tqdm_insts = []
def __thread_clear_event():
    global __clear_event, __cleared_tqdm_insts, __write_lock
    while True:
        __clear_event.wait()
        with __write_lock:
            __cleared_tqdm_insts = __tqdm_clear_all_instance()

def __thread_refresh_event():
    global __refresh_event, __cleared_tqdm_insts, __write_lock
    while True:
        __refresh_event.wait()
        with __write_lock:
            __tqdm_refresh_all(__cleared_tqdm_insts)
        __cleared_tqdm_insts = []

def lock_packing():
    global __write_lock, __clear_event, __refresh_event
    return (__write_lock, __clear_event, __refresh_event)

def print(*args, **kwargs):
    args = [a if isinstance(a, str) else str(a) for a in args]
    __print(kwargs.get('sep', ' ').join(args))

def __print(string, file=None, end="\n", nolock=False):
    """Print a message via tqdm (without overlap with bars)."""
    fp = file if file is not None else sys.stdout
    with __tqdm_external_write_mode(file=file, nolock=nolock):
        # Write the message
        fp.write(string)
        fp.write(end)
        #fp.flush()
        #time.sleep(1)

def __tqdm_clear_all_instance():
    fp = sys.stdout

    # Clear all bars
    inst_cleared = []
    for inst in getattr(tqdm.tqdm, '_instances', []):
        # Clear instance if in the target output file
        # or if write output + tqdm output are both either
        # sys.stdout or sys.stderr (because both are mixed in terminal)
        if hasattr(inst, "start_t") and (inst.fp == fp or all(
                f in (sys.stdout, sys.stderr) for f in (fp, inst.fp))):
            inst.refresh(nolock=True)
            inst.clear(nolock=True)
            inst_cleared.append(inst)
    
    #fp.flush()
    #time.sleep(1)
    return inst_cleared

def __tqdm_refresh_all(insts):
    fp = sys.stdout

    # Force refresh display of bars we cleared
    for inst in insts:
        inst.refresh(nolock=True)
    
    #fp.flush()
    #time.sleep(1)

@contextmanager
def __tqdm_external_write_mode(file=None, nolock=False):
    """
    Disable tqdm within context and refresh tqdm when exits.
    Useful when writing to standard output stream
    """
    global __clear_event, __refresh_event
    fp = file if file is not None else sys.stdout

    try:
        if not nolock:
            tqdm.tqdm.get_lock().acquire()
        __clear_event.set()
        inst_cleared = __tqdm_clear_all_instance()
        yield
        __refresh_event.set()
        __tqdm_refresh_all(inst_cleared)
    finally:
        if not nolock:
            tqdm.tqdm._lock.release()

def dummy_fn(device, tqdm_position, *args):
    """Example Pool Inner Function. 
    The experiment function should follows this format.

    Args:
        device (int): device id
        tqdm_position (int): tqdm position.
            Use like this `tqdm.tqdm(position=tqdm_poisiton)`

    Returns:
        Any: the return value of experiment. This should be pickleable.
    """
    import random

    # doing some job while reporting tqdm
    for i in tqdm.tqdm(range(1000), desc=f'dev:{device}, args:{args}', position=tqdm_position):
        time.sleep(1/500)
        if (i % 100) == -1:
            print(f"Runner@{device}: {random.choice(['Hello World!', 'Dooby is free', 'Shana', 'Ivan Polka'])}")
    time.sleep(random.random() * 3)
    
    # may job raise random exception
    if random.random() < 0.1:
        raise Exception('random exception')
    
    # job could return output with out of order
    return random.randint(0, 1000)

def runtime_wrapper(ret_queue: "mp.Queue", tqdm_lock: "mp.RLock", fn, device, tqdm_position, lock_package, *args):
    tqdm.tqdm.set_lock(tqdm_lock)
    initialize(*lock_package)

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
        traceback_text = traceback.format_exc()
        print(traceback_text)
        ret_queue.put({
            'status': 'failed',
            'ex': ex,
            'args': args,
            'args_string': args_text,
            'device': device,
            'tqdm_position': tqdm_position,
            'traceback': traceback_text
        })

class GPUPool:
    def __init__(self, devices=None):
        if devices is None:
            devices = query_available_devices()
            print('GPUPool: Available devices,', devices)
        self.devices = devices
        self.retries = 5
        self.queue = None

        if not 'RLock' in str(tqdm.tqdm.get_lock()):
            print('GPUPool: tqdm write lock is not mp.RLock. Initialized.')
            tqdm.tqdm.set_lock(mp.RLock())
        
        if not is_initialized():
            print('GPUPool: tqdm service is not initizlied. Initialize.')
            initialize()
    
    def run(self, fn, args_list: list):
        """
        Run args list with parallel devices.
        This function is out of order.
        """

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
                        self.devices.index(target_device), lock_packing(), *args ),
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
                try:
                    if proc['p'].is_alive():
                        proc['p'].kill()
                except:
                    print('error while clean up process')
                    traceback.print_exc()

if __name__ == '__main__':
    initialize()
    pool = GPUPool(devices=list(range(4)))
    
    ret = pool.run(dummy_fn, list(range(18)))
    assert len(ret) == 18
    print('Pool result:', ret)

    # the pool object could be reusable, because always new process is created when run called.
    ret = pool.run(dummy_fn, list(range(18)))
    assert len(ret) == 18
    print('Pool result:', ret)