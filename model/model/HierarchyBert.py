from numpy.core.defchararray import not_equal
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import BertModel,RobertaModel
from transformers import AutoModel
from tools.accuracy_tool import multi_label_accuracy, single_label_top1_accuracy

import random

class HierarchyBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(HierarchyBert, self).__init__()

        # self.encoder = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        # self.encoder = AutoModel.from_pretrained("thunlp/Lawformer")
        self.encoder = AutoModel.from_pretrained("/data/disk1/private/xcj/MJJDInfoExtract/SimilarCase/code/LegalRoBERTa")
        labels = json.load(open(config.get('data', 'label2num'), 'r'))
        min_data_num = 10
        self.class_num = len([l for l in labels if labels[l] >= min_data_num]) + 1
        self.class_num2 = len(set([l.split('/')[0] + '/' + l.split('/')[1] for l in labels if labels[l] >= min_data_num])) + 1

        self.hidden_size = 768
        weight = torch.ones(self.class_num).float()
        weight[0] = 0.3
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        weight2 = torch.ones(self.class_num2).float()
        weight2[0] = 0.3
        self.criterion2 = nn.CrossEntropyLoss(weight=weight2)
        self.accuracy_function = single_label_top1_accuracy
        self.fc = nn.Linear(self.hidden_size * 10, self.class_num)
        self.fc2 = nn.Linear(self.hidden_size * 10, self.class_num2)
        self.elu = nn.ELU()
        self.pooler = nn.Linear(self.hidden_size * 10, self.hidden_size * 10)
        self.softmax = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()
        self.criterion3 = nn.CrossEntropyLoss()

        self.positive_score = nn.Linear(self.hidden_size * 10, 1)

    def init_multi_gpu(self, device, config, *args, **params):
        return

    def loss_para_space(self, mapmat):
        # mapmat: label3num, label2num
        # self.fc.weight: label3num, hiddensize * 10
        # self.fc2.weight: label2num, hiddensize * 10
        map2mat = mapmat.matmul(self.fc2.weight) # label3num, hiddensize * 10
        diff = map2mat - self.fc.weight
        loss = torch.sum(diff * diff) * 0.5
        return loss

    def loss_output_space(self, mapmat, output2, output3):
        # mapmat: label3num, label2num
        # output2: batch, label2num
        # output3: batch, label3num
        score2 = self.softmax(output2)
        score3 = self.softmax(output3)
        score23 = output2.matmul(torch.transpose(mapmat, 0, 1))
        diff = self.relu(score3 - score23) # batch, label3num
        loss = diff.mean()
        return loss

    def forward(self, data, config, gpu_list, acc_result, mode):
        if mode != "train":
            return self.forward_test(data, config, gpu_list, acc_result, mode)
        inputx = data['input']
        batch = inputx.shape[0]

        output = self.encoder(inputx, attention_mask=data['mask'])#, global_attention_mask = data["gAtt"])
        clses = output['last_hidden_state'][:,:10,:] # batch, 10, hidden_size
        clses = self.elu(self.pooler(self.elu(clses.view(batch, 10 * self.hidden_size)))) # batch, 10 * hidden_size

        score3 = self.fc(clses).view(-1, self.class_num) # batch * (neg+1), class_num
        if mode == 'train':
            score2 = self.fc2(clses).view(-1, self.class_num2) # batch * (neg+1), class_num2
            loss = self.criterion(score3, data["label"]) + self.criterion2(score2, data["label2"]) 
            loss = loss + self.loss_para_space(data['map']) * 1e-5
            loss = loss + self.loss_output_space(data["map"], score2, score3) * 1e-5

            pos_score = self.positive_score(clses).view(-1, 5)
            # loss = loss + 0.5 * self.criterion3(pos_score, torch.zeros(pos_score.shape[0], dtype=torch.long).to(pos_score.device))

            acc_result = accuracy(score3, pos_score, data["label"], config, acc_result)
            return {"loss": loss, "acc_result": acc_result}
        else:
            acc_result = accuracy_doc(score3, data["label"], config, acc_result)
            # acc_result = accuracy_doc_top(score3, data["label"], config, acc_result)
            return {"loss": 0, "acc_result": acc_result}

    def forward_test(self, data, config, gpu_list, acc_result, mode):
        inputx = data['input']
        batch = inputx.shape[0]

        allscore = []
        posscore = []
        max_num_per_batch = 8
        begin = 0
        while begin < batch:
            end = min(begin + max_num_per_batch, batch)
            output = self.encoder(inputx[begin: end], attention_mask=data['mask'][begin: end])#, global_attention_mask = data["gAtt"][begin: end])
            clses = output['last_hidden_state'][:,:10,:] # batch, 10, hidden_size
            clses = self.elu(self.pooler(self.elu(clses.view(end - begin, 10 * self.hidden_size)))) # batch, 10 * hidden_size

            score3 = self.fc(clses).view(-1, self.class_num) # batch * (neg+1), class_num
            pscore = self.positive_score(clses)
            allscore.append(score3)
            posscore.append(pscore)
            begin = end

        acc_result = accuracy_doc_top(torch.cat(allscore, dim = 0), data["label"], config, acc_result)
        # acc_result = accuracy_doc_pos(torch.cat(allscore, dim = 0), torch.cat(posscore, dim = 0), data["label"], config, acc_result)
        return {"loss": 0, "acc_result": acc_result}

def accuracy_doc_pos(score, poscore, label, config, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0, 'pos_acc': 0, 'pos_num': 0, "labelset": 0, "doc_num": 0}
    pre_score, pre_res = torch.max(score, dim=1)
    # predict = set(pre_res.tolist()) - {0}
    predict = set()
    positive_res = (-poscore).argsort()# [:poscore.shape[0] // 2]
    for pos in positive_res:
        if pre_res[pos] != 0:
            predict.add(int(pre_res[pos]))
        if label[pos] != 0:
            acc_result["pos_acc"] += 1

    lset = set(label.tolist()) - {0}

    acc_result['labelset'] += len(predict)
    acc_result['doc_num'] += 1

    acc_result['actual_num'] += len(lset)
    acc_result['pre_num'] += len(predict)
    acc_result['right'] += len(lset & predict)
    acc_result["pos_num"] += len(lset)

    return acc_result

def accuracy_doc_top(score, label, config, acc_result):
    # score: para_num, class_num
    # label: para_num
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0, 'labelset': 0, 'doc_num': 0}
    # score_p = torch.softmax(score, dim = 1)
    # pre_score, pre_res = torch.max(score_p, dim=1)
    # predict = set()
    # pre_score, pre_res = pre_score.tolist(), pre_res.tolist()
    # for s, l in zip(pre_score, pre_res):
    #     if s > 0.8 and l != 0:
    #         predict.add(int(l))
    pre_score, pre_res = torch.max(score, dim=1)
    predict = set(pre_res.tolist()) - {0}

    # if len(predict) < 2:
    #     lmaxscore = torch.max(score, dim = 0)[0]
    #     labels = (-lmaxscore).argsort().tolist()
    #     for l in labels:
    #         if len(predict) >= 2 or l == 0 or l in predict:
    #             continue
    #         predict.add(l)

    lset = set(label.tolist()) - {0}

    acc_result['labelset'] += len(predict)
    acc_result['doc_num'] += 1

    acc_result['actual_num'] += len(lset)
    acc_result['pre_num'] += len(predict)
    acc_result['right'] += len(lset & predict)

    return acc_result


def accuracy_doc(score, label, config, acc_result):
    # score: para_num, class_num
    # label: para_num
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0, 'labelset': 0, 'doc_num': 0}

    pre_res = torch.max(score, dim = 1)[1] # para_num
    predict = set(pre_res.tolist()) - {0} # merges.argsort()[:3].tolist()

    if len(predict) == 0:
        score[:,0] -= 1000
        tscore = torch.max(score, dim = 0)[0]
        pre = torch.max(tscore, dim = 0)[1]
        predict.add(pre)
    # predict = set()
    # tscore = torch.max(torch.softmax(score, dim=1), dim=0)[0].tolist()
    '''
    tindex = (-tscore).argsort().tolist()
    now = 0
    for ind in tindex:
        if ind == 0:
            continue
        s = float(tscore[ind])
        if now < 1 and len(predict) <= 3:
            predict.add(ind)
            now += s
        else:
            break
    '''
    # for index, s in enumerate(tscore):
    #     if s > 0.2:
    #         predict.add(index)

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

def accuracy(score, posscore, label, config, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'pre_num': 0, 'actual_num': 0, "pos_right": 0, "pos_num": 0}
    predict = torch.max(score, dim=1)[1]
    acc_result['pre_num'] += int((predict != 0).int().sum())
    acc_result['right'] += int((predict[label != 0] == label[label != 0]).int().sum())
    acc_result['actual_num'] += int((label != 0).int().sum())

    acc_result["pos_num"] += int(posscore.shape[0])
    acc_result["pos_right"] += int((torch.max(posscore, dim = 1)[1] == 0).sum())
    return acc_result


def accuracy_per_label(score, label, config, acc_result):
    # score : para_num, class_num
    if acc_result is None:
        acc_result = [{'right': 0, 'pre_num': 0, 'actual_num': 0} for i in range(len(label.shape[1]))]
    pre_score, pre_res = torch.max(score, dim=1)
    llist = set(label.tolist())
    for l in llist:
        if l == 0:
            continue
        acc_result[l]["actual_num"] += 1
    plist = set(pre_res.tolist())
    for l in plist:
        if l == 0:
            continue
        acc_result[l]["pre_num"] += 1
    for r in (llist & plist):
        if r == 0:
            continue
        acc_result[r]["right"] += 1
    return acc_result