import math
import queue, time
import threading
import torch
import numpy as np
import transformers
from datasets import load_dataset
from utils.process_pool import ProcessPool, BatchIterator

class ImagesHfDataset:
    def __init__(self, 
        train_transform, test_transform, 
        batch_size=4, num_workers_train=16, num_workers_test=4, 
        name='food101', split='train[:5000]', test_split='split'
    ):
        self.train_pool = ProcessPool(num_workers_train, train_transform)
        self.test_pool = ProcessPool(num_workers_test, test_transform)

        self.data = load_dataset(name, split=split, cache_dir='./cache/datasets')
        if test_split == 'split':
            self.data = self.data.shuffle(seed=42).train_test_split(test_size=0.1)
            self.train_set = self.data['train']
            self.test_set = self.data['test']
        else:
            self.test_data = load_dataset(name, split=test_split, cache_dir='./cache/datasets')
            self.train_set = self.data
            self.test_set = self.test_data
        
        if 'labels' in self.train_set.features:
            self.dataset_labels_tag = 'labels'
        elif 'label' in self.train_set.features:
            self.dataset_labels_tag = 'label'
        elif 'fine_label' in self.train_set.features:
            self.dataset_labels_tag = 'fine_label'
        else:
            raise Exception('label not found', self.train_set.features)
        if self.dataset_labels_tag != 'labels':
            self.test_set = self.test_set.rename_column(self.dataset_labels_tag, 'labels')
            self.train_set = self.train_set.rename_column(self.dataset_labels_tag, 'labels')

        if 'image' in self.train_set.features:
            self.dataset_image_tag = 'image'
        elif 'images' in self.train_set.features:
            self.dataset_image_tag = 'images'
        elif 'img' in self.train_set.features:
            self.dataset_image_tag = 'img'
        else: raise Exception('image not found', self.train_set.features)
        if self.dataset_image_tag != 'image':
            self.test_set = self.test_set.rename_column(self.dataset_image_tag, 'image')
            self.train_set = self.train_set.rename_column(self.dataset_image_tag, 'image')

        if 'coarse_label' in self.train_set.features:
            self.train_set = self.train_set.remove_columns(['coarse_label'])
            self.test_set = self.test_set.remove_columns(['coarse_label'])

        self.labels = self.train_set.features['labels'].names
        self.num_labels = len(self.labels)
        self.id2label = {str(i): c for i, c in enumerate(self.labels)}
        self.label2id = {c: str(i) for i, c in enumerate(self.labels)}
        
        self.batch_size = batch_size
    
    def get_train_iter(self):
        return BatchIterator(self.train_set.shuffle(), self.train_pool, self.batch_size)

    def get_test_iter(self):
        return BatchIterator(self.test_set, self.test_pool, self.batch_size)

class ExamplesToBatchTransform:
    def __init__(self, item_transform):
        self.transform = item_transform
    
    def __call__(self, examples):
        examples = [self.transform(ex) for ex in examples]
        names = examples[0].keys()
        ret = {}
        for name in names:
            items = []
            is_label = False
            for ex in examples:
                i = ex[name]
                if isinstance(i, int):
                    items.append(i)
                    is_label = True
                elif isinstance(i, torch.Tensor):
                    items.append(i)
                else:
                    raise Exception('unsupported type', type(i))
            if is_label:
                ret[name] = torch.tensor(items)
            else: ret[name] = torch.stack(items, dim=0)
        return ret

class ViTInputTransform:
    def __init__(self, extractor: "transformers.ViTFeatureExtractor", test=False):
        self.extractor = extractor
        
        from torchvision import transforms
        if test:
            self.transform = transforms.Compose([
                transforms.Resize((extractor.size, extractor.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(extractor.size), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomApply(torch.nn.ModuleList([
                    transforms.ColorJitter(brightness=.5, hue=.3),
                    transforms.Grayscale(3),
                    transforms.GaussianBlur(3),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAdjustSharpness(3),
                    # transforms.RandomRotation(degrees=(0, 180)),
                    # transforms.RandomPosterize(bits=4)
                ]), p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
            ])
    
    def __call__(self, example):
        example['pixel_values'] = self.transform(example['image'].convert('RGB'))
        if 'label' in example:
            example['labels'] = example['label']
            del example['label']
        del example['image']
        return example

if __name__ == '__main__':
    import tqdm
    transform = ExamplesToBatchTransform(ViTInputTransform(transformers.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")))
    data = ImagesHfDataset(transform, transform, name='cifar100', split='train', test_split='test', batch_size=64)
    for item in data.get_train_iter():
        print(item)
        break
    for ix in range(10):
        print('epoch', ix)
        for i, item in enumerate(tqdm.tqdm(data.get_train_iter())):
            item = {k: item[k].to(0, non_blocking=True) for k in item.keys()}
