import multiprocessing as mp
import concurrent.futures

class ThreadBuffer:
    def __init__(self, use_thread=True):
        if use_thread:
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        else:
            self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        self.function = None
        self.result = None
        self.lastArgs = None
        self.lastFunction = None
    
    def checkRefresh(self, function, args):
        if self.result is None or self.lastArgs is None or self.lastFunction is None:
            return True
        if self.lastFunction is function:
            return False
        if len(self.lastArgs) is len(args):
            for i in range(len(args)):
                if not(self.lastArgs[i] is args[i]):
                    return True
            return False
        return True
    
    def queue(self, function, args):
        self.result = self.pool.submit(function, *args)
        self.lastArgs = args
        self.lastFunction = function

    def get(self, function, args):
        if self.checkRefresh(function, args):
            self.queue(function, args)
        ret = self.result.result()
        del self.result
        self.queue(function, args)
        return ret

    def close(self):
        self.pool.shutdown()