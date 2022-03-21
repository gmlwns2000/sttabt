import transformers
from torchtext.datasets import WikiText103
from utils import ThreadBuffer
import random
import multiprocessing as mp

class FilteredWikitext:
    def __init__(self, min_length = 50):
        self.data = None
        self.length = 0
        self.min_length = min_length
        for i in self:
            self.length += 1
    
    def __iter__(self):
        self.data = iter(WikiText103(split='train'))
        return self
    
    def __next__(self):
        line = ""
        while len(line) < self.min_length:
            line = next(self.data)
        return line
    
    def __len__(self):
        return self.length

class WikitextBatchLoader:
    def __init__(self, batch_size, tokenizer):
        self.data = FilteredWikitext()
        self.bank = []
        for i in self.data:
            self.bank.append(i)
        self.tokenizer = None
        self.batch_size = batch_size
        self.buffer = ThreadBuffer()
        self.index = 0
        self.queue = mp.Queue(maxsize=64)
        self.procs = []
        self.num_workers = 4
        for i in range(self.num_workers):
            proc = mp.Process(target=self.worker_main, daemon=True)
            proc.start()
            self.procs.append(proc)
    
    def worker_main(self):
        print('WikitextBatchLoader: worker_main')
        while True:
            item = self.random_batch()
            self.queue.put(item)

    def random_sample(self):
        #mimic GLUE
        line = self.bank[random.randint(0, len(self.bank) - 1)].strip()
        #random cut
        spl = line.split()
        if len(spl) > 10:
            spl = spl[:random.randint(10,len(spl))]
        line = ' '.join(spl)
        #mimic cls
        if random.random() < 0.75:
            line = "[CLS]"+line
        #mimic sep
        for i in range(random.randint(0, 3)):
            if random.random() > 0.5:
                if random.random() > 0.5:
                    spl = line.split()
                    spl.insert(random.randint(0, len(spl)-1), "[SEP]")
                    line = ' '.join(spl)
                else:
                    pivot = random.randint(0, len(line)-1)
                    line = line[:pivot] + "[SEP]" + line[pivot:]
        if random.random() < 0.75:
            line = line + "[SEP]"
        return line

    def random_batch(self):
        lines = [self.random_sample() for i in range(self.batch_size)]
        if self.tokenizer is None:
            self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        result = self.tokenizer(lines, padding=True, truncation=True, max_length=512, return_tensors='pt')
        item = {
            'input_ids': result.input_ids,
            'attention_mask': result.attention_mask,
        }
        return item
    
    def __iter__(self):
        self.index = 0
        return self

    def ___next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        self.index += 1
        return self.queue.get()

    def __next__(self):
        return self.___next__()
    
    def __len__(self):
        return len(self.bank) // self.batch_size

if __name__ == '__main__':
    import transformers
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = WikitextBatchLoader(16, tokenizer)
    for i in range(50):
        print(data.random_sample())