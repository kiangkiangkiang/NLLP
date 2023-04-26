import numpy as np
import pandas as pd

data_root = "/Users/cfh00892302/Desktop/myWorkspace/NLLP/data_preprocess/"
random_size = 5

my_final_data = np.load(data_root + "final_data.npy", allow_pickle=True)
jasper_data = np.loadtxt(data_root + "test_for_jasper.txt", dtype=str)
ind = np.random.choice(range(len(my_final_data)), random_size, replace=False)
print(ind)

for i in ind:
    print("This JID:", my_final_data[i][0][0])
    print(my_final_data[i][6][0][:70])
    print("====================")

my_final_data[4288][6][0]

all_file[0][0][0]

import json

result = []
with open("/Users/cfh00892302/Desktop/myWorkspace/NLLP/data/toy_data.json", encoding="utf-8") as f:
    tmp = json.loads(f.readline())
    for i in tmp:
        result.append(i)

result[3]["jid"][:].replace("\u3000", "")
