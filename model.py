import copy
import torch.nn as nn
from sklearn.model_selection import train_test_split

class AraT5_PMTL(nn.Module):
    '''
        decoder split level:
        1: two separate decoders for each task
        2: two separate lm_heads for each task
        
    '''
    def __init__(self, model, decoder_split_level = 2):
        super(AraT5_PMTL, self).__init__()
        self.model = model

        if decoder_split_level == 2:
            self.lm_head_abstractive = copy.deepcopy(self.model.lm_head)
            self.lm_head_task2 = copy.deepcopy(self.model.lm_head)
        elif decoder_split_level == 1:
            self.decoder_abstractive = copy.deepcopy(self.model.decoder)
            self.decoder_task2 = copy.deepcopy(self.model.decoder)

        self.decoder_split_level = decoder_split_level

    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None, task=None):
        if self.decoder_split_level == 2:
            self.model.lm_head = self.lm_head_task2 if task == "task2" else self.lm_head_abstractive
        elif self.decoder_split_level == 1:
            self.model.decoder = self.decoder_task2 if task == "task2" else self.decoder_abstractive
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            # decoder_attention_mask=decoder_attention_mask,
        )
        
    def generate(self, **kwargs):
        task = kwargs.get("task", None)
        kwargs.pop("task", None)

        if self.decoder_split_level == 2:
            self.model.lm_head = self.lm_head_task2 if task == "task2" else self.lm_head_abstractive
        elif self.decoder_split_level == 1:
            self.model.decoder = self.decoder_task2 if task == "task2" else self.decoder_abstractive

        return self.model.generate(**kwargs)