import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def output_function1(data, config, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0
    

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    if "pos_right" in data and data["pos_num"] != 0:
        pos_acc = data["pos_right"] / data["pos_num"]
        metric["pos_acc"] = round(pos_acc, 4)
    if 'labelset' in data and 'doc_num' in data and data['doc_num'] != 0:
        metric['ave_len'] = data['labelset'] / data['doc_num']
    return json.dumps(metric)

def output_perlabel(data, config, *args, **params):
    for i in range(1, len(data)):
        d = data[i]
        if d["pre_num"] != 0 and d["actual_num"] != 0:
            data[i]["precision"] = d["right"] / d["pre_num"]
            data[i]["recall"] = d["rigth"] / d["actual_num"]
        else:
            data[i]["precision"] = 0
            data[i]["recall"] = 0
    fout = open("result.txt", "a")
    ret = json.dumps(data)
    print(ret, file=fout)
    fout.close()
    return ret
