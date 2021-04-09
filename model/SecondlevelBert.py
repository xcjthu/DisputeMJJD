import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging

logger = logging.getLogger(__name__)

from transformers import BertModel,RobertaModel

from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy

class SecondlevelBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SecondlevelBert, self).__init__()

        self.encoder = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        secondlabels = list(set([l.split('/')[0] + '/' + l.split('/')[1] for l in labels if labels[l] >= 20]))
        self.class_num = len(secondlabels) + 1 # 加上‘无标签’

        self.hidden_size = 768
        weight = torch.ones(self.class_num).float()
        weight[0] = 0.3 #这个weight是干嘛用的？
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        # self.accuracy_function = single_label_top1_accuracy
        self.fc = nn.Linear(self.hidden_size, self.class_num)
    def init_multi_gpu(self, device, config, *args, **params):
        return

    def forward(self, data, config, gpu_list, acc_result, mode):
        inputx = data['input']

        # _, bcls = self.encoder(inputx, attention_mask=data['mask'])
        res = self.encoder(inputx, attention_mask=data['mask'])

        result = self.fc(res['pooler_output']).view(-1, self.class_num)

        # batch * (neg+1), class_num 输出层是否在某个阈值范围内，如（0-1）;加入sigmoid
        if mode == 'train':
            loss = self.criterion(result, data["secondlabel"])

            # acc_result = self.accuracy_function(result, data["label"], config, acc_result)
            # acc_result = accuracy(result, data["secondlabel"], config, acc_result)
            acc_result = accuracy_topK(result, data["secondlabel"], config, acc_result)

            return {"loss": loss, "acc_result": acc_result}
        else:
            acc_result = accuracy_topK(result, data["secondlabel"], config, acc_result)
            return {"loss": 0, "acc_result": acc_result}

def accuracy_topK(score, label, config, acc_result):
    K = 0.9  # K为阈值
    # 正确率根据阈值放水，标签为空也算一种标签。
    # score = nn.functional.sigmoid(score)
    score = torch.sigmoid(score)

    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0}

    predict = torch.gt(score, K) #取得阈值大于K的作为输出标签
    pred_tag = 0
    batch_num = 0
    predict_ind_list = []
    for line in predict.tolist(): #获得阈值为True的元素的下标(有没有简单的写法啊)
        tmp = []
        batch_num += 1
        for ind,el in enumerate(line):
            if el == True:
                tmp.append(ind)
                pred_tag += 1
        predict_ind_list.append(tmp)
    print("average tags:%.3f"%(pred_tag/len(predict_ind_list)))

    pre_num = 0
    for el in predict_ind_list:
        if len(el):
            pre_num += 1

    right_num = 0
    for id, el in enumerate(label): #计算正确的数量（只要存在即正确）
        if el.int() in predict_ind_list[id]:
            right_num += 1

    acc_result['pre_num'] += pre_num
    acc_result['right'] += right_num
    acc_result['actual_num'] += len(label)
    return acc_result

def accuracy_doc(score, label, config, acc_result):
    # score: para_num, class_num
    # label: para_num
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0, 'labelset': 0, 'doc_num': 0}
    '''
    pre_res = torch.max(score, dim = 1)[1] # para_num
    predict = set(pre_res.tolist()) - {0} # merges.argsort()[:3].tolist()
    '''
    '''
    if len(predict) == 0:
        score[:,0] -= 1000
        tscore = torch.max(score, dim = 0)[0]
        pre = torch.max(tscore, dim = 0)[1]
        predict.add(pre)
    '''
    predict = set()
    tscore = torch.max(torch.softmax(score, dim=1), dim=0)[0].tolist()
    for index, s in enumerate(tscore):
        if s > 0.15:
            predict.add(index)
    predict = predict - {0}
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
