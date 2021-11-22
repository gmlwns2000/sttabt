import random

from torchtext.datasets import DATASETS
import transformers, torch, tqdm

from dataset.classification_batch_entry import ClassificationBatchEntry

class ClassificationDataset:
    def __init__(self, 
        batch_size = 4,
        tokenizer = 'electra',
        dataset = 'AG_NEWS'
    ):
        self.batch_size = batch_size
        self.dataset_name = dataset
        self.num_classes = 0
        self.tokenizer = transformers.ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
        
        def proc(set):
            data = []
            max_len = 0
            avg_len = 0
            for item in tqdm.tqdm(set):
                idx, text = item

                max_len = max(max_len, len(text))
                avg_len += len(text)
                self.num_classes = max(self.num_classes, idx + 1)
                
                data.append((idx, text,))
            print(max_len, avg_len / len(data))
            return data
        
        train_set, test_set = DATASETS[dataset](root='./cache')
        self.train_set = proc(train_set)
        self.test_set = proc(test_set)
    
    def batch(self, test=False):
        labels = []
        texts = []

        for i in range(self.batch_size):
            set = self.train_set
            if test: set = self.test_set
            idx, text = set[random.randint(0, len(set)-1)]
            labels.append(idx)
            texts.append(text)

        output = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

        entry = ClassificationBatchEntry()
        entry.labels = torch.tensor(labels, dtype=torch.int64)
        entry.input_ids = output['input_ids']
        entry.attention_masks = output['attention_mask']
        entry.raw_texts = texts
        return entry

if __name__ == '__main__':
    data = ClassificationDataset(batch_size=64)

    for _ in range(100):
        batch = data.batch()
        #print(batch.raw_texts)
        print(batch.input_ids.shape)