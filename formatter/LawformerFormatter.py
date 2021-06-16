from transformers import AutoTokenizer
import json
import torch
import os
import numpy as np

from formatter.Basic import BasicFormatter


class LawformerFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode
        self.max_len = config.getint("train", "max_len")

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.label2id = json.load(open(config.get("data", "label2id")))

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = np.zeros((len(data), len(self.label2id)))

        for did, doc in enumerate(data):
            tokens = self.tokenizer.encode(doc["input"], truncation=True, max_length=self.max_len, add_special_tokens=True)
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens += [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))

            inputx.append(tokens)
            for l in doc['label']:
                if l not in doc["label"]:
                    print(l)
                label[did,self.label2id[l]] = 1
        gatt = np.zeros((len(data), self.max_len))
        gatt[:, 0] = 1
        return {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "gAtt": torch.tensor(gatt, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

