from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs
from mmcv import Config, DictAction
import sys
import os
from tqdm import tqdm
'''
cfg.dataset_root = 'data/nuscenes/'
cfg.train_pipeline[4].db_sampler.dataset_root = 'data/nuscenes/'
cfg.train_pipeline[4].db_sampler.info_path = 'data/nuscenes/nuscenes_dbinfos_train.pkl'
cfg.train_pipeline[7].dataset_root = 'data/nuscenes/'
cfg.test_pipeline[6].dataset_root = 'data/nuscenes/'
cfg.data.train.dataset.dataset_root ='data/nuscenes/'
cfg.data.train.dataset.ann_file = 'data/nuscenes/nuscenes_infos_train.pkl'
cfg.data.train.dataset.pipeline[4].db_sampler.dataset_root = 'data/nuscenes/'
cfg.data.train.dataset.pipeline[4].db_sampler.info_path = 'data/nuscenes/nuscenes_dbinfos_train.pkl'
cfg.data.train.dataset.pipeline[7].dataset_root = 'data/nuscenes/'
cfg.data.val.dataset_root = 'data/nuscenes/'
cfg.data.val.ann_file = 'data/nuscenes/nuscenes_infos_val.pkl'
cfg.data.val.pipeline[6].dataset_root = 'data/nuscenes/'
cfg.data.test.dataset_root = 'data/nuscenes/'
cfg.data.test.ann_file = 'data/nuscenes/nuscenes_infos_val.pkl'
cfg.data.test.pipeline[6].dataset_root = 'data/nuscenes/'
cfg.evaluation.pipeline[6].dataset_root = 'data/nuscenes/'
'''
# sys.path.append("/datav/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion")
'''
root@d150de2f8d32:/datav/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion# tree configs/
configs/
|-- default.yaml
`-- nuscenes
    |-- default.yaml
    |-- det
    |   |-- centerhead
    |   |   |-- default.yaml
    |   |   `-- lssfpn
    |   |       |-- camera
    |   |       |   |-- 256x704
    |   |       |   |   |-- default.yaml
    |   |       |   |   `-- swint
    |   |       |   |       `-- default.yaml
    |   |       |   `-- default.yaml
    |   |       `-- default.yaml
    |   |-- default.yaml
    |   `-- transfusion
    |       |-- default.yaml
    |       `-- secfpn
    |           |-- camera+lidar
    |           |   |-- default.yaml
    |           |   `-- swint_v0p075
    |           |       |-- convfuser.yaml
    |           |       `-- default.yaml
    |           |-- default.yaml
    |           `-- lidar
    |               |-- default.yaml
    |               |-- pointpillars.yaml
    |               |-- voxelnet.yaml
    |               `-- voxelnet_0p075.yaml
    `-- seg
        |-- camera-bev256d2.yaml
        |-- default.yaml
        |-- fusion-bev256d2-lss.yaml
        `-- lidar-centerpoint-bev128.yaml

13 directories, 21 files

'''

base_root = "/datav/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/"

# 1. 设置convfuser.yaml 路径
args_config = '/datav/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'

# 2. 迭代加载配置文件
configs.load(args_config, recursive=True)
cfg = Config(recursive_eval(configs), filename=args_config)
# print(type(cfg)) # <class 'mmcv.utils.config.Config'>

# 3. 创建dataset
dataset = build_dataset(cfg.data.test)

## 3.1 test.py 情况下build_dataset的输入数据
# /datav/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/configs/nuscenes/default.yaml
"""
{'type': 'NuScenesDataset',  
'dataset_root': 'data/nuscenes/', 
'ann_file': 'data/nuscenes/nuscenes_infos_val.pkl', 
'pipeline': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...],
'object_classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', ...],  
'map_classes': ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider'], 
'modality': {'use_lidar': True, 'use_camera': True, 'use_radar': False, 'use_map': False, 'use_external': False}, 
'test_mode': True, 
'box_type_3d': 'LiDAR'}
"""

# 4. dataset构建完成后，第一条数据的key
# print(dataset) # <mmdet3d.datasets.nuscenes_dataset.NuScenesDataset object at 0x7f7fd58d1af0>
# for i in dataset:
#     for j in i:
#         print(j)
#     break

# for i in dataset:
#     print(i)
#     break

"""
# img
# points
# gt_bboxes_3d 
# gt_labels_3d
# gt_masks_bev
# camera_intrinsics
# camera2ego
# lidar2ego
# lidar2camera
# camera2lidar
# lidar2image
# img_aug_matrix
# lidar_aug_matrix
# metas
 """

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=True,
    shuffle=False,
)

print(len(data_loader))

pbar = tqdm(data_loader)
for index, data in enumerate(pbar):
    print(data)
    pbar.set_description = f"{index}"
    break
    
    

