import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        self.data = json.load(open(self.data_path, "r"))
        for i in range(len(self.data)):
            label = []
            for j in range(len(self.data[i]["label"])):
                if len(self.data[i]["label"][j].split("/")) != 3:
                    continue
                self.data[i]["label"][j] = self.data[i]["label"][j].replace("\n", "").replace(" ", "").replace("规则：\"", "")
                if self.data[i]["label"][j][:3] == "间借贷":
                    self.data[i]["label"][j] = "民" + self.data[i]["label"][j]
                if self.data[i]["label"][j][-1] == "." or self.data[i]["label"][j][-1] == "。":
                    self.data[i]["label"][j] = self.data[i]["label"][j][:-1]
                label.append(self.data[i]["label"][j])
            self.data[i]["label"] = label

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
