import json
import os
from torch.utils.data import Dataset
import numpy as np
import random

class ParaPosDataset(Dataset):
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
            if len(labels) > 0:
                self.data.append(docs[did])
        self.paras = []
        for docid, doc in enumerate(self.data):
            for pid, para in enumerate(doc):
                para['id'] = '%s_%s' % (docid, pid)
                self.paras.append(para)
        self.paras = [p for p in self.paras if len(p['label']) > 0]
        para_len = np.array([len(p['para']) for p in self.paras])
        print('==' * 10, mode, 'dataset', '==' * 10)
        print('label num:', len(self.label2id))
        print('doc num:', len(self.data), 'para num:', len(self.paras))
        print('average para len', para_len.mean(), 'max para len', para_len.max())
        print('%s paras are longer than the max len' % ((para_len > self.max_len).sum() / len(self.paras)))
        print('==' * 25)

    def __getitem__(self, item):
        return self.paras[item]

    def __len__(self):
        return len(self.paras)
