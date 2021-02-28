import json
import xlrd
import os

if __name__ == "__main__":
    for filename in os.listdir("."):
        if not filename.endswith("xlsx"):
            continue
        excel = xlrd.open_workbook(filename)
        table = excel.sheets()[0]

        keys = ['争议焦点', '裁判观点', '裁判依据', '说理', '判例']
        zyjd = None
        rules = {}
        for rowid in range(1, table.nrows):
            line = list(table.row_values(rowid))
            if line[1] == line[2] == line[3] == line[4] == '':
                zyjd = line[0][2:]
                rules[zyjd] = []
                continue
            ldata = {keys[i]: line[i] for i in range(5)}
            rules[zyjd].append(ldata)

        json.dump(rules, open(filename.replace("xlsx", "json"), 'w', encoding="utf8"), indent=2, ensure_ascii=False)
