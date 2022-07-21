import math
import queue, time
import threading
import torch.multiprocessing as mp
import traceback
import torch
import numpy as np

mp.set_sharing_strategy('file_system')

class ProcessPool:
    def __init__(self, num_worker, func) -> None:
        self.func = func
        self.job_queue = mp.Queue(maxsize=num_worker*3)
        self.return_queue = mp.Queue(maxsize=num_worker*3)
        self.worker_finished = -1
        self.worker_ready = 0
        self.num_worker = num_worker
        self.procs = []
        self.thread = None

        self.start_proc()
        self.start_fetch()

    def start_proc(self):
        procs = []
        for ip in range(self.num_worker):
            proc = mp.Process(target = self.proc_main, daemon=True, args=(ip,))
            proc.start()
            procs.append(proc)
        self.procs = procs
    
    def proc_main(self, id):
        torch.set_num_threads(1)
        while True:
            item = self.job_queue.get()
            if item == 'EOF':
                self.return_queue.put('EOF')
                #wait for empty queue
                #print('received EOF', id)
                while not self.job_queue.empty():
                    time.sleep(0.001)
                while not self.return_queue.empty():
                    time.sleep(0.001)
                self.return_queue.put('READY')
                #print('emit ready', id)
            else:
                ret = self.func(item)
                self.return_queue.put(ret)

    def start_fetch(self):
        self.fetch_queue = queue.Queue()
        self.thread = threading.Thread(target=self.fetch_main, daemon=True)
        self.thread.start()

    def fetch_main(self):
        accum_errors = 10
        while True:
            try:
                item = self.return_queue.get()
            except RuntimeError as ex:
                traceback.print_exc()
                print('Pool.FetchThread: error while get return_queue', ex)
                accum_errors -= 1
                if accum_errors < 0:
                    raise ex
                else:
                    time.sleep(0.1)
                    continue
            if item == 'EOF':
                self.worker_finished += 1
                if self.worker_finished >= self.num_worker:
                    self.ready_all_workers()
                    self.fetch_queue.put(None)
            elif item == 'READY':
                self.worker_ready += 1
            else:
                self.fetch_queue.put(item)
    
    def ready_all_workers(self):
        for _ in range(self.num_worker):
            item = self.return_queue.get()
            if item == 'READY':
                self.worker_ready += 1
            if self.worker_ready >= self.num_worker:
                self.worker_ready = 0
                break

    def close(self):
        for proc in self.procs:
            proc.kill()
        self.job_queue.close()
        self.return_queue.close()

    def reset_eof(self):
        assert self.worker_finished == -1 or self.worker_finished == self.num_worker
        if self.worker_finished == -1:
            self.send_eof()
            while True:
                ret = self.fetch_queue.get()
                if ret is None:
                    break
                else:
                    time.sleep(1e-3)
        self.worker_finished = -1
        self.worker_ready = 0
        while not self.job_queue.empty():
            try:
                self.job_queue.get(timeout=0.1)
            except: pass

    def send_eof(self):
        self.worker_finished = 0
        for _ in range(self.num_worker):
            self.job_queue.put('EOF')

    def push(self, item):
        if item == 'EOF':
            self.send_eof()
        else:
            self.job_queue.put(item)

    def get(self):
        return self.fetch_queue.get()

class BatchIterator:
    def __init__(self, items, pool: ProcessPool, batch_size):
        self.items = items
        self.pool = pool
        self.count = len(items)
        self.batch_size = batch_size

        self.idx = 0
        self.ended = False
        self.pool.reset_eof()
    
    def __len__(self):
        return math.ceil(self.count / self.batch_size)
    
    def __iter__(self):
        self.idx = 0
        self.ended = False
        self.pool.reset_eof()
        return self

    def __next__(self):
        #push items
        try:
            while (not self.pool.job_queue.full()) and (self.idx < self.count):
                items = []
                for i in range(self.batch_size):
                    if self.idx < self.count:
                        item = self.items[self.idx]
                        self.idx += 1
                        items.append(item)
                    else:
                        break
                
                if len(items) > 0: 
                    self.pool.push(items)
                
            if self.idx >= self.count and (not self.ended):
                self.ended = True
                self.pool.push('EOF')
                #print('pushed EOF')
            
            #print('POOL STAT', self.pool.job_queue.qsize(), self.idx)
            batch = self.pool.get()
            if batch is None:
                raise StopIteration
            return batch
        except KeyboardInterrupt:
            import traceback
            traceback.print_exc()
            self.pool.close()
            raise KeyboardInterrupt