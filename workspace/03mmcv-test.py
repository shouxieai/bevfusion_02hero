import mmcv
import json

data = mmcv.load("data/nuscenes/nuscenes_infos_val.pkl")

for i in data:
    print(i)
    for j in data[i]:
        print(j)
        break
    break