import json

path = "/Users/cfh00892302/Desktop/myWorkspace/NLLP/data/"
file_name = "toy_data.json"
text = []
with open(path + file_name, "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        text.append(json.loads(line))


len(text)
len(text[0])
