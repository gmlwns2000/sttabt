from transformers import BertModel
from torch import nn
from torch.nn import functional as F
import torch

from dataset.classification_batch_entry import ClassificationBatchEntry

class BertClassification(nn.Module):
    def __init__(self, n_classes, bert_model_name = 'google/bert_uncased_L-4_H-256_A-4'):
        super().__init__()

        self.bert_model_name = bert_model_name
        self.bert = BertModel.from_pretrained(self.bert_model_name)
        self.hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, n_classes),
        )

    def forward(self, batch:ClassificationBatchEntry, return_output=False):
        lm_output = self.bert(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_masks, 
            output_hidden_states=True
        )
        lm_output = lm_output.last_hidden_state[:,0,:]
        x = self.classifier(lm_output)
        
        loss = F.cross_entropy(x, batch.labels)
        if not return_output: return loss
        return loss, x