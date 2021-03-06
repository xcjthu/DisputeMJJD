import json
import torch
import os

from formatter.Basic import BasicFormatter


class CharFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.max_len = config.getint("data", "max_seq_length")
        self.word2id = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))
        self.mode = mode
        self.mapping = json.load(open(config.get("data", "mapping_file"), "r", encoding="utf8"))
        self.multi = config.getboolean("data", "multi")
        self.label2num = json.load(open('/data/disk1/private/xcj/MJJDInfoExtract/SimilarCase/code/xcjshr/data/court_view/label2num.json', 'r'))

    def tokenize(self, text):
        res = []
        for c in text:
            if c in self.word2id.keys():
                res.append(self.word2id[c])
            else:
                res.append(self.word2id["[UNK]"])
        res = [self.word2id["[CLS]"]] + res
        while len(res) < self.max_len:
            res.append(self.word2id["[PAD]"])
        res = res[:self.max_len]

        return res

    def process(self, data, config, mode, *args, **params):
        input = []
        if mode != "test":
            label = []

        for temp in data:
            text = temp["content"]
            token = self.tokenize(text)

            input.append(token)
            if mode != "test":
                if self.multi:
                    label.append([])
                    for a in range(0, len(self.mapping["id2name"])):
                        name = self.mapping["id2name"][str(a)]
                        if name in set(temp["label"]):
                            label[-1].append(1)
                        else:
                            label[-1].append(0)
                else:
                    label.append(self.mapping["name2id"][temp["label"][0]])

        input = torch.LongTensor(input)
        if mode != "test":
            label = torch.LongTensor(label)
            biaslabel = label.float()
            biaslabel[biaslabel==1] -= 0.1
            biaslabel[biaslabel==0] += 0.1

        if mode != "test":
            return {'input': input, 'label': label, 'biaslabel': biaslabel}
        else:
            return {"input": input}
