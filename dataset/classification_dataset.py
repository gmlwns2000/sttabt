import random

from torchtext.datasets import DATASETS
import transformers, torch, tqdm

from dataset.classification_batch_entry import ClassificationBatchEntry

LABELS = {
    'AG_NEWS': {0: 'World',
                1: 'Sports',
                2: 'Business',
                3: 'Sci/Tech'},
    'SogouNews': {0: 'Sports',
                  1: 'Finance',
                  2: 'Entertainment',
                  3: 'Automobile',
                  4: 'Technology'},
    'DBpedia': {0: 'Company',
                1: 'EducationalInstitution',
                2: 'Artist',
                3: 'Athlete',
                4: 'OfficeHolder',
                5: 'MeanOfTransportation',
                6: 'Building',
                7: 'NaturalPlace',
                8: 'Village',
                9: 'Animal',
                10: 'Plant',
                11: 'Album',
                12: 'Film',
                13: 'WrittenWork'},
    'YelpReviewPolarity': {0: 'Negative polarity',
                           1: 'Positive polarity'},
    'YelpReviewFull': {0: 'score 1',
                       1: 'score 2',
                       2: 'score 3',
                       3: 'score 4',
                       4: 'score 5'},
    'YahooAnswers': {0: 'Society & Culture',
                     1: 'Science & Mathematics',
                     2: 'Health',
                     3: 'Education & Reference',
                     4: 'Computers & Internet',
                     5: 'Sports',
                     6: 'Business & Finance',
                     7: 'Entertainment & Music',
                     8: 'Family & Relationships',
                     9: 'Politics & Government'},
    'AmazonReviewPolarity': {0: 'Negative polarity',
                             1: 'Positive polarity'},
    'AmazonReviewFull': {0: 'score 1',
                         1: 'score 2',
                         2: 'score 3',
                         3: 'score 4',
                         4: 'score 5'}
}

class ClassificationDataset:
    def __init__(self, 
        batch_size = 4,
        tokenizer = 'electra',
        dataset = 'YahooAnswers'
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
            print(f'Classification Dataset Stat.: name:{self.dataset_name}, nclass:{self.num_classes}, max_len:{max_len}, avg_len:{avg_len / len(data)}, count:{len(data)}')
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
            text = f'[CLS]{text}'
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
    data = ClassificationDataset(batch_size=4)

    for _ in range(100):
        batch = data.batch()
        #print(batch.raw_texts)
        print(batch.input_ids.shape)