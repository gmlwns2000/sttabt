class ClassificationBatchEntry:
    #inputs
    raw_texts = None
    input_ids = None
    attention_masks = None

    #labels
    labels = None

    #device
    device = None

    def to(self, device):
        self.device = device

        if not self.input_ids is None: self.input_ids = self.input_ids.to(device)
        if not self.attention_masks is None: self.attention_masks = self.attention_masks.to(device)
        if not self.labels is None: self.labels = self.labels.to(device)