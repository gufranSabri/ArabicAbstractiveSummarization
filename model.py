import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import copy
import pandas as pd
from torch.utils.data import DataLoader
from data import MultiTaskDataset
from sklearn.model_selection import train_test_split


class AraT5_PMTL(nn.Module):
    def __init__(self, model):
        super(AraT5_PMTL, self).__init__()
        self.model = model
        self.lm_head_abstractive = copy.deepcopy(self.model.lm_head)
        self.lm_head_extractive = copy.deepcopy(self.model.lm_head)

    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None, task=None):
        self.model.lm_head = self.lm_head_extractive if task == "qa" else self.lm_head_abstractive

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        
    def generate(self, **kwargs):
        task = kwargs.get("task", None)
        kwargs.pop("task", None)

        self.model.lm_head = self.lm_head_extractive if task == "qa" else self.lm_head_abstractive
        return self.model.generate(**kwargs)
    

if __name__ == "__main__":
    model_name = "UBC-NLP/AraT5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, resume_download=True)

    model = AraT5_PMTL(model)
    print(model)