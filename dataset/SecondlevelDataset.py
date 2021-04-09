import json
import os
from torch.utils.data import Dataset
import numpy as np
import random

class SecondlevelDataset(Dataset):
    """
    secondlabel为二级标签内容；
    label为三级标签内容；
    """
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")

        docs = json.load(open(config.get('data', "%s_data_path" % mode), 'r'))
        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        self.label2id = {'NA': 0}
        # add second level label
        second_labels = set()
        for l in labels:
            if labels[l] >= 20:
                self.label2id[l] = len(self.label2id)
                second_labels.add(l.split('/')[0]+'/'+l.split('/')[1])
        self.second_label2id = {'NA': 0}
        second_labels = list(second_labels)
        for id, s in enumerate(second_labels):
            self.second_label2id[s] = id+1

        self.data = []
        for did, doc in enumerate(docs):
            labels = []
            second_labels = []
            for pid, para in enumerate(doc):
                docs[did][pid]['label'] = [l for l in para['label'] if l in self.label2id]
                # wsc: add second level label
                docs[did][pid]['secondlabel'] = [l.split('/')[0]+'/'+l.split('/')[1] for l in para['label'] if l.split('/')[0]+'/'+l.split('/')[1] in self.second_label2id]
                labels += docs[did][pid]['label']
                second_labels += docs[did][pid]['secondlabel']
            if len(labels) > 0 or mode == 'test':
                self.data.append(docs[did])

        self.paras = []
        for docid, doc in enumerate(self.data):
            for pid, para in enumerate(doc):
                para['id'] = '%s_%s' % (docid, pid)
                if len(para['para']) < 512:
                    self.paras.append(para)
                else:
                    sents = para['para'].split('。') # 对于大于512的doc进行截断
                    for beg in range(0, len(sents), 4):
                        self.paras.append({'para': '。'.join(sents[beg : beg + 8]), 'label': para['label'], 'secondlabel': para['secondlabel'], 'id': para['id']})
        self.pos_paras = [p for p in self.paras if len(p['label']) > 0]
        self.neg_paras = [p for p in self.paras if len(p['label']) == 0]
        para_len = np.array([len(p['para']) for p in self.paras])
        print('==' * 10, mode, 'dataset', '==' * 10)
        print('label num:', len(self.label2id))
        # print('len of candidate', len(self.good_para), 'drop pos:', drop_pos_num, 'neg: ', drop_neg_num)
        print('doc num:', len(self.data), 'para num:', len(self.paras))
        print('positive num:', len(self.pos_paras), 'negative num:', len(self.neg_paras))
        print('average para len', para_len.mean(), 'max para len', para_len.max())
        print('%s paras are longer than the max len' % ((para_len > self.max_len).sum() / len(self.paras)))
        print('==' * 25)

    def __getitem__(self, item):
        if self.mode == 'train':
            return [self.pos_paras[item]] + random.sample(self.neg_paras, 3)
        else:
            return self.data[item]

    def __len__(self):
        if self.mode == 'train':
            return len(self.pos_paras)
        else:
            return len(self.data)
