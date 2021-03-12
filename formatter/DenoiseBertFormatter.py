from transformers import BertTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random

class DenoiseBertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    def process_test(self, data):
        inputx = []
        mask = []
        for para in data:
            tokens = self.tokenizer.encode(para['para'], max_length=self.max_len, add_special_tokens=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
        return {
            'inputx': torch.LongTensor(inputx),
            'mask': torch.LongTensor(mask),
            'ids': [para['id'] for para in data],
        }

    def process(self, data, config, mode, *args, **params):
        inputx = []
        neginputx = []
        mask = []
        negmask = []
        if mode == 'test':
            return self.process_test(data)

        for paras in data:
            tokens = self.tokenizer.encode(paras['pos']['para'], max_length=self.max_len, add_special_tokens=True)
            negtokens = self.tokenizer.encode(paras['neg']['para'], max_length=self.max_len, add_special_tokens=True)

            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            negmask.append([1] * len(negtokens) + [0] * (self.max_len - len(negtokens)))

            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            negtokens += [self.tokenizer.pad_token_id] * (self.max_len - len(negtokens))

            inputx.append(tokens)
            neginputx.append(negtokens)

        return {
            'inputx': torch.LongTensor(inputx),
            'neginputx': torch.LongTensor(neginputx),
            'mask': torch.LongTensor(mask),
            'negmask': torch.LongTensor(negmask),
            'label': torch.zeros(len(data)).long(),
            }
