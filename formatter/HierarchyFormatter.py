from transformers import BertTokenizer,RobertaTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random

class HierarchyFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        self.label2id = {'NA': 0}
        for l in labels:
            if labels[l] >= 10:
                self.label2id[l] = len(self.label2id)

        self.label2id_2 = {'NA': 0}
        self.label32id2 = {'NA': 0}

        for l in self.label2id:
            if l == 'NA':
                continue
            key = l.split('/')
            key = key[0] + '/' + key[1]
            if key not in self.label2id_2:
                self.label2id_2[key] = len(self.label2id_2)
            self.label32id2[l] = self.label2id_2[key]

        self.prefix = self.tokenizer.convert_tokens_to_ids(['[CLS]'] * 10)

        self.map = np.zeros((len(self.label2id), len(self.label2id_2)))
        self.map[0,0] = 1
        for l in self.label2id:
            if l == 'NA':
                continue
            key = l.split('/')
            key = key[0] + '/' + key[1]
            self.map[self.label2id[l],self.label2id_2[key]] = 1

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        label_2 = []
        for paras in data:
            for para in paras:
                if len(para['label']) == 0:
                    label.append(self.label2id['NA'])
                    label_2.append(self.label2id_2['NA'])
                else:
                    label.append(self.label2id[random.choice(para['label'])])
                    label_2.append(self.label32id2[random.choice(para['label'])])
                tokens = self.tokenizer.encode(para['para'], max_length=self.max_len - 11, add_special_tokens=False, truncation=True)
                tokens = self.prefix + tokens + [self.tokenizer.sep_token_id]
                mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
                tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
                inputx.append(tokens)
        gatt = np.zeros((len(inputx), self.max_len))
        gatt[:, :10] = 1
        return {
            'input': torch.LongTensor(inputx),
            'mask': torch.LongTensor(mask),
            'label': torch.LongTensor(label),
            'label2': torch.LongTensor(label_2),
            'map': torch.FloatTensor(self.map),
            'gAtt': torch.LongTensor(gatt),
        }
