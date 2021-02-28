import json
import os

input_path = ["../data/origin.json", "../data/origin2.json"]
output_path = "../data/mapping.json"

if __name__ == "__main__":
    nameset = set()
    nameset.add("无适用焦点")
    for filename in input_path:
        data = json.load(open(filename, "r", encoding="utf8"))
        for item1 in data["task"]["options"]:
            name1 = item1["value"]
            if "children" not in item1.keys():
                continue
            for item2 in item1["children"]:
                name2 = item2["value"]
                for item3 in item2["children"]:
                    name3 = item3["value"]
                    nameset.add(name1 + "/" + name2 + "/" + name3)

    mapping = {"name2id": {"无适用焦点": 0}, "id2name": ["无适用焦点"], "metainf": {"无适用焦点": {}}}

    for filename in os.listdir("rules"):
        if filename.endswith("json"):
            data = json.load(open(os.path.join("rules", filename), "r", encoding="utf8"))
            for name in data.keys():
                for item in data[name]:
                    s = name + "/" + item["争议焦点"] + "/" + item["裁判观点"]
                    if s in mapping["name2id"].keys():
                        continue
                    if s not in nameset:
                        continue
                    mapping["name2id"][s] = len(mapping["name2id"])
                    mapping["id2name"].append(s)
                    mapping["metainf"][s] = item

    json.dump(mapping, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)
