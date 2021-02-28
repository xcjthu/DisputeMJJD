import json

input_path = "/data/disk1/private/xcj/MJJDInfoExtract/SimilarCase/data/rules/rules.json"
output_path = "../data/mapping.json"

if __name__ == "__main__":
    data = json.load(open(input_path, "r", encoding="utf8"))

    mapping = {"name2id": {}, "id2name": [], "metainf": {}}
    for name in data.keys():
        for item in data[name]:
            mapping["name2id"][name + "/" + item["争议焦点"] + "/" + item["裁判观点"]] = len(mapping["name2id"])
            mapping["id2name"].append(name + "/" + item["争议焦点"] + "/" + item["裁判观点"])
            mapping["metainf"][name + "/" + item["争议焦点"] + "/" + item["裁判观点"]] = item

    json.dump(mapping, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)
