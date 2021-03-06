import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy


class ParaBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(ParaBert, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        self.class_num = len([l for l in labels if labels[l] >= 20]) + 1

        self.hidden_size = 768
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = single_label_top1_accuracy
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        inputx = data['input']

        _, bcls = self.encoder(inputx, attention_mask=data['mask'])
        result = self.fc(bcls).view(-1, self.class_num) # batch * (neg+1), class_num
        if mode == 'train':
            loss = self.criterion(result, data["label"])
            # acc_result = self.accuracy_function(result, data["label"], config, acc_result)
            acc_result = accuracy(result, data["label"], config, acc_result)
            return {"loss": loss, "acc_result": acc_result}
        else:
            acc_result = accuracy_doc(result, data["label"], config, acc_result)
            return {"loss": 0, "acc_result": acc_result}

def accuracy_doc(score, label, config, acc_result):
    # score: para_num, class_num
    # label: para_num
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0, 'labelset': 0, 'doc_num': 0}

    pre_res = torch.max(score, dim = 1)[1] # para_num
    predict = set(pre_res.tolist()) - {0} # merges.argsort()[:3].tolist()
    '''
    predict = set()
    pre_res = score.argsort()[:2].tolist()
    for res in pre_res:
        if res[0] == 0:
            continue
        else:
            predict.add(res[0])
            predict.add(res[1])
    predict = predict - {0}
    '''
    lset = set(label.tolist()) - {0}
    assert len(lset) != 0
    #print(predict, lset)
    acc_result['actual_num'] += len(lset)
    acc_result['pre_num'] += len(predict)
    acc_result['right'] += len(lset & predict)
    acc_result['labelset'] += len(predict)
    acc_result['doc_num'] += 1
    return acc_result

def accuracy(score, label, config, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0}
    predict = torch.max(score, dim=1)[1]
    acc_result['pre_num'] += int((predict != 0).int().sum())
    acc_result['right'] += int((predict[label != 0] == label[label != 0]).int().sum())
    acc_result['actual_num'] += int((label != 0).int().sum())
    return acc_result
