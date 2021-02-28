import json
import os

input_list = ["../../data/origin.json", "../../data/origin2.json"]
output_path = "../../data/sentence/all.json"
mapping = json.load(open("../../data/mapping.json", "r", encoding="utf8"))

if __name__ == "__main__":
    result = []
    cnt = 0

    for input_path in input_list:
        data = json.load(open(input_path, "r", encoding="utf8"))

        print(len(data["term"]))

        for item in data["term"]:
            for a in range(0, len(item["content"][0]["content"])):
                s = item["content"][0]["content"][a]
                if s.find("\n") != -1:
                    continue
                temp = {
                    "content": s,
                    "label": "无适用焦点",
                    "id": str(item["termID"]) + "_" + str(a)
                }

                able = True
                for x in item["result"][0]["answer"]:
                    if x["index"] == a:
                        temp["label"] = x["value"][0]
                        if temp["label"] not in mapping["name2id"].keys():
                            cnt += 1
                            able = False
                            # print(temp["label"])
                if not able:
                    continue

                result.append(temp)

    print(cnt)
    print(len(result))
    json.dump(result, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
