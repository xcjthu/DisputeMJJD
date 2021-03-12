import json
import os
from torch.utils.data import Dataset
import numpy as np
import random

class DenoiseDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")

        docs = json.load(open(config.get('data', "%s_data_path" % mode), 'r'))
        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        self.label2id = {'NA': 0}
        for l in labels:
            if labels[l] >= 20:
                self.label2id[l] = len(self.label2id)
        self.data = []
        for did, doc in enumerate(docs):
            labels = []
            for pid, para in enumerate(doc):
                docs[did][pid]['label'] = [l for l in para['label'] if l in self.label2id]
                labels += docs[did][pid]['label']
            if len(labels) > 0 or mode == 'test':
                self.data.append(docs[did])
        self.paras = []
        for docid, doc in enumerate(self.data):
            for pid, para in enumerate(doc):
                para['id'] = '%s_%s' % (docid, pid)
                if len(para['para']) < 512:
                    self.paras.append(para)
                else:
                    sents = para['para'].split('。')
                    for beg in range(0, len(sents), 4):
                        self.paras.append({'para': '。'.join(sents[beg : beg + 8]), 'label': para['label'], 'id': para['id']})
        self.pos_paras = [p for p in self.paras if len(p['label']) > 0]
        self.neg_paras = [p for p in self.paras if len(p['label']) == 0]
        para_len = np.array([len(p['para']) for p in self.paras])
        print('==' * 10, mode, 'dataset', '==' * 10)
        print('label num:', len(self.label2id))
        print('doc num:', len(self.data), 'para num:', len(self.paras))
        print('positive num:', len(self.pos_paras), 'negative num:', len(self.neg_paras))
        print('average para len', para_len.mean(), 'max para len', para_len.max())
        print('%s paras are longer than the max len' % ((para_len > self.max_len).sum() / len(self.paras)))
        print('==' * 25)

    def __getitem__(self, item):
        if self.mode != 'test':
            return {
                'pos': self.pos_paras[item],
                'neg': random.choice(self.neg_paras),
            }
        else:
            return self.paras[item]

    def __len__(self):
        if self.mode != 'test':
            return len(self.pos_paras)
        else:
            return len(self.paras)
