import json
import random

input_path = "../../data/document/all.json"
train_file = "../../data/document/train.json"
test_file = "../../data/document/test.json"

if __name__ == "__main__":
    train = []
    test = []
    for item in json.load(open(input_path, "r", encoding="utf8")):
        if random.randint(1, 5) == 1:
            test.append(item)
        else:
            train.append(item)

    json.dump(train, open(train_file, "w", encoding="utf8"), ensure_ascii=False, sort_keys=True, indent=2)
    json.dump(test, open(test_file, "w", encoding="utf8"), ensure_ascii=False, sort_keys=True, indent=2)
