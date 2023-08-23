import pickle
import mmcv
from mmdet3d.core.utils.visualize import visualize_camera
import os
import os


current_file_path = os.path.abspath(__file__)
meta_path = os.path.join(current_file_path[:-15], "data", "metas.pkl")
result_path = os.path.join(current_file_path[:-15], "data", "results.pkl")

with open(meta_path, 'rb') as f:
    metas = pickle.load(f)

with open(result_path, 'rb') as f:
    outputs = pickle.load(f)

metas = metas[0]
name = "{}-{}".format(metas["timestamp"], metas["token"])
bboxes = outputs[0][0].to("cpu")
scores = outputs[0][1]
labels = outputs[0][2]
bboxes.corners[..., 2] -= bboxes.tensor.data[..., 5].reshape(-1, 1)

# 替换路径
file_name_new = [os.path.join(current_file_path[:-15], "data", item.split("/")[-1]) for item in metas["filename"]]


for k, image_path in enumerate(file_name_new):
    image = mmcv.imread(image_path)
    visualize_camera(
        os.path.join(current_file_path[:-15], "vis", f"camera-{k}", f"{name}.png"),
        image,
        bboxes=bboxes,
        labels=labels,
        transform=metas["lidar2image"][k],
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
    )