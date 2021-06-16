import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoModel
from .loss import MultiLabelSoftmaxLoss
from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class Lawformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Lawformer, self).__init__()

        self.encoder = AutoModel.from_pretrained("thunlp/Lawformer")
        self.class_num = len(json.load(open(config.get("data", "label2id"))))

        self.hidden_size = 768
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = MultiLabelSoftmaxLoss(config, self.class_num)
        self.accuracy_function = multi_label_accuracy
        self.fc = nn.Linear(self.hidden_size, self.class_num * 2)

    def forward(self, data, config, gpu_list, acc_result, mode):
        output = self.encoder(data['inputx'], attention_mask=data['mask'], global_attention_mask = data["gAtt"])
        bcls = output["pooler_output"]
        result = self.fc(bcls).view(-1, self.class_num, 2)

        loss = self.criterion(result, data["label"])
        acc_result = acc(result, data["label"], acc_result)

        return {"loss": loss, "acc_result": acc_result}

def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0}
    pred = (score[:,:,0] < score[:,:,1]).int()
    acc_result["right"] += int(pred[label == 1].sum())
    acc_result["pre_num"] += int(pred.sum())
    acc_result["actual_num"] += int(label.sum())


    return acc_result

