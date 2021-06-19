from transformers import BertTokenizer,RobertaTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random

class ParaBertPosFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        self.label2id = {'NA': 0}
        for l in labels:
            if labels[l] >= 20:
                self.label2id[l] = len(self.label2id)


    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        for para in data:
            if len(para['label']) == 0:
                label.append(self.label2id['NA'])
            else:
                label.append(self.label2id[random.choice(para['label'])])
            tokens = self.tokenizer.encode(para['para'], max_length=self.max_len, add_special_tokens=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            inputx.append(tokens)
        return {
            'input': torch.LongTensor(inputx),
            'mask': torch.LongTensor(mask),
            'label': torch.LongTensor(label),
        }
