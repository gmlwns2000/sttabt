from transformers import BigBirdModel
from torch import nn
from torch.nn import functional as F
import torch

from dataset.classification_batch_entry import ClassificationBatchEntry

class BigBirdClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.bird = BigBirdModel.from_pretrained('google/bigbird-roberta-base')
        self.hidden = self.bird.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, n_classes),
        )

    def forward(self, batch:ClassificationBatchEntry, return_output=False):
        lm_output = self.bird(input_ids = batch.input_ids, attention_mask = batch.attention_masks, output_hidden_states=True)
        lm_output = lm_output.last_hidden_state[:,0,:]
        x = self.classifier(lm_output)
        
        loss = F.cross_entropy(x, batch.labels)
        if not return_output: return loss
        return loss, x