import json
import os

input_list = ["../../data/origin.json", "../../data/origin2.json"]
output_path = "../../data/document/all.json"


def format_content(content, id=0):
    pos = []
    for a in range(0, len(content)):
        if content[a] == "\n\n":
            pos.append(a)
    p1 = 1
    while pos[p1] - pos[p1 - 1] <= 2:
        p1 += 1
    p2 = len(pos) - 1
    while pos[p2] - pos[p2 - 1] <= 2:
        p2 -= 1

    content = content[pos[p1]:pos[p2]]
    res = []
    for x in content:
        if x.find("\n") == -1:
            res.append(x.strip())

    return "".join(res)


if __name__ == "__main__":
    result = []

    for input_path in input_list:
        data = json.load(open(input_path, "r", encoding="utf8"))

        print(len(data["term"]))

        for item in data["term"]:
            temp = {
                "content": format_content(item["content"][0]["content"], item["termID"]),
                "label": [],
                "id": item["termID"]
            }
            for x in item["result"][0]["answer"]:
                temp["label"].append(x["value"][0])
            temp["label"] = list(set(temp["label"]))

            result.append(temp)

    json.dump(result, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
