import json

input_path = "../../data/document/all.json"
min_freq = 5
word2id = "../../data/document/word2id.json"
mapping = json.load(open("../../data/mapping.json", "r", encoding="utf8"))

cnt = {}
label_cnt = []
for a in range(0, len(mapping["name2id"])):
    label_cnt.append(0)

if __name__ == "__main__":
    total_len = 0
    total = 0
    max_len = 0
    min_len = 100000
    for item in json.load(open(input_path, "r", encoding="utf8")):
        total_len += len(item["content"])
        min_len = min(min_len, len(item["content"]))
        max_len = max(max_len, len(item["content"]))
        total += 1
        for x in item["content"]:
            if x not in cnt.keys():
                cnt[x] = 0
            cnt[x] += 1

        for a in range(0, len(label_cnt)):
            if mapping["id2name"][a] in set(item["label"]):
                label_cnt[a] += 1

    print(total_len / total)
    print(min_len, max_len)
    print(len(cnt))
    dic = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
    for c in cnt.keys():
        if cnt[c] >= min_freq:
            dic[c] = len(dic)
    json.dump(dic, open(word2id, "w", encoding="utf8"), indent=2, ensure_ascii=False)
    print(len(dic))

    print(len(label_cnt))
    for num in range(0, 5):
        x = 0
        for a in label_cnt:
            if a <= num:
                x += 1
        print(num, x)
