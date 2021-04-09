from transformers import BertTokenizer,RobertaTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter
import random

class SecondlevelBertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        self.label2id = {'NA': 0}
        # add second level label
        second_labels = set()
        for l in labels:
            if labels[l] >= 20:
                self.label2id[l] = len(self.label2id)
                second_labels.add(l.split('/')[0] + '/' + l.split('/')[1])
        self.second_label2id = {'NA': 0}
        second_labels = list(second_labels)
        for id, s in enumerate(second_labels):
            self.second_label2id[s] = id

        '''
        self.label2id_2 = {'NA': 0}
        for l in self.label2id:
            if l == 'NA':
                continue
            key = l.split('/')
            key = key[0] + '/' + key[1]
            if key not in self.label2id_2:
                self.label2id_2[key] = len(self.label2id_2)
            self.label2id[l] = self.label2id_2[key]
        '''
    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        secondlabel = []
        for paras in data:
            for para in paras:
                if len(para['secondlabel']) == 0:
                    secondlabel.append(self.second_label2id['NA'])
                else:
                    secondlabel.append(self.second_label2id[random.choice(para['secondlabel'])]) # 此处random的做法是否合理
                tokens = self.tokenizer.encode(para['para'], max_length=self.max_len, add_special_tokens=True)
                tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
                mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
                inputx.append(tokens)
        return {
            'input': torch.LongTensor(inputx),
            'mask': torch.LongTensor(mask),
            'secondlabel': torch.LongTensor(secondlabel),
            }
