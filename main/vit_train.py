import torch
import numpy as np
import transformers
from datasets import load_dataset

def main():
    vit_extractor = transformers.ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    vit_model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    data = load_dataset("food101", split="train[:5000]", cache_dir='./cache/datasets')
    data = data.train_test_split(test_size=0.2)
    labels = data["train"].features["label"].names
    
    from torchvision.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip, Normalize, ToTensor
    data_aug = Compose([
        RandomResizedCrop(vit_extractor.size), 
        RandomHorizontalFlip(), 
        ToTensor(), 
        Normalize(mean=vit_extractor.image_mean, std=vit_extractor.image_std)
    ])
    def transforms(examples):
        examples['pixel_values'] = [data_aug(img.convert('RGB')) for img in examples['image']]
        del examples['image']
        return examples
    data = data.with_transform(transforms)

    data_collator = transformers.DefaultDataCollator()

    training_args = transformers.TrainingArguments(
        output_dir="./saves/vit-baseline/",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = transformers.Trainer(
        model=vit_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=vit_extractor,
    )
    
    trainer.train()

if __name__ == '__main__':
    main()