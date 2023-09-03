from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion # 四元数操作的包
from nuscenes.utils.data_classes import Box # 10.1 创建Box
import numpy as np
import cv2
import os
import inspect

# 1. 实例化NuScenes
version = "v1.0-mini"  # 因为我们是用的是mini数据集
dataroot = "/app/121BEVFusion/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/data/nuscenes" # 我的数据集放这里了
nuscenes = NuScenes(version=version, dataroot=dataroot, verbose=False)
# print(len(nuscenes.sample)) # 404

sample = nuscenes.sample[0]

# 2. 获取lidar的数据
lidar_sample_token = sample["data"]["LIDAR_TOP"] # 拿到data中LIDAR_TOP的sample_token 9d9bf11fb0e144c8b446d54a8a00184f
lidar_sample_data = nuscenes.get('sample_data', lidar_sample_token)

lidar_filename = os.path.join(dataroot, lidar_sample_data["filename"]) # samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin
lidar_point_cloud_data = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 5)
print(f"行号[{inspect.currentframe().f_lineno}]:\n",lidar_point_cloud_data.shape) # (34688, 5)

'''
3. 坐标系
    3.1 全局坐标系，global coordinate
        - 可以简单的认为，车辆在t0试课的位置认为是全局坐标系的原点  
    3.2 车体坐标系，ego_pose. ego coordinate
        - 以车体为原点的坐标系
    3.3 传感器坐标系
        - lidar   的坐标系
        - camera  的坐标系
        - radar   的坐标系

4. 标定calibater
lidar的标定，获得的结果是：lidar相对于ego而言的位置(translation)，和旋转(rotation) 
    - translation 可以用3个float数字表示位置
        - 相对于ego而言的位置
    - rotation则是用4个float表示旋转，用的是四元数
    
camera的标定，获得的结果是：camera相对于ego而言的位置 (translation)，和旋转(rotation) 
    - translation 可以用3个float数字表示位置
        - 相对于ego而言的位置
    - rotation则是用4个float表示旋转，用的是四元数
    - carmera 还多了一个camera_intrinsic 相机的内参（3d->2d平面）
    - 相机畸变参数（目前nuScenes数据集不考虑）

5. 坐标系转换
yxy:理解为，ego相当于中间坐标系
    - 点云数据属于lidar坐标系，
    - 如果我想把点云的数据转换到camera的坐标系。
        - 就需要先把点云 从lidar坐标系，
        - 转换到ego坐标系，再从ego坐标系转换到camera坐标系

5.1 但是转换存在一个难点。---不同传感器的频率不同，也就是说捕获数据的时间不同。
    意味着：lidar和camera的起始时刻可能不一致，起始时刻对应的ego_pose也不一样
        lidar  捕获的timestamp是t0，t0 -> ego_pose0   t0对应着一个车的姿态信息
        camera 捕获的timestamp是t1，t1 -> ego_pose1   t1对应着一个车的姿态信息
        
    因此，为了考虑到时间timestamp的问题，需要使用全局坐标系，global coordinate
    此时，如果想将进行如下点云转换到image上，即lidar_points -> image 
        - 过程为 timestamp = t0 时的lidar_points -> ego_pose0 -> global -> ego_pose1 -> camera -> intrinsic -> image
        - 即 t0时刻点云信息 转换到 t0时刻车体坐标系上 -> 全局坐标系 -> t1时刻车体坐标系上 -> t1时刻相机坐标系 -> 通过相机内参将数据转换到 -> image上
'''

'''
6. 上面5.1 中的过程对应代码 lidar_points -> ego_pose0 -> global -> ego_pose1 -> camera -> intrinsic -> image
    6.1 timestamp = t0 时的lidar_points -> ego_pose0
         6.1.1) lidar_points 对应 lidar coordinate。 ego_pose 对应 ego coordinate
            能进行转换，意味着存在变换矩阵，将 雷达点云  转换到ego坐标系上。
            下方代码演示如何得到变换矩阵
            
''' 
def get_matrix(calibrated_data, inverse=False):
    """
    args:
    @calibrated_data : calibrated_sensor对象。一般通过nuscenes.get("calibrated_sensor"，token..)得到
    @inverse : 是否取逆矩阵。
    具体根据calibrated_sensor对象里面的 rotation 与 translation 计算出一个4*4的旋转平移矩阵。
    如果inverse设置为ture。则对这个矩阵逆变换
    """
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

lidar_calibrated_data = nuscenes.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

# 6.1.2 
# lidar_to_ego_matrix 是基于ego而言的。
# point = lidar_to_ego_matrix @ lidar_points.T   代表了lidar -> ego 的过程。
lidar_to_ego_matrix = get_matrix(lidar_calibrated_data)
print(f"行号[{inspect.currentframe().f_lineno}]:\n",lidar_to_ego_matrix) 
'''
[[ 0.00203327  0.99970406  0.02424172  0.943713  ]
 [-0.99998053  0.00217566 -0.00584864  0.        ]
 [-0.00589965 -0.02422936  0.99968902  1.84023   ]
 [ 0.          0.          0.          1.        ]]
'''

# 6.2 timestamp = t0 时的ego_pose0 -> global
# ego_to_global_matrix 是基于ego而言的。
# point = ego_to_global_matrix @ lidar_points.T
ego_pose_data0 = nuscenes.get("ego_pose", lidar_sample_data["ego_pose_token"])
# print(ego_pose_data0)
'''
{'token': '9d9bf11fb0e144c8b446d54a8a00184f', 
'timestamp': 1532402927647951, 
'rotation': [0.5720320396729045, -0.0016977771610471074, 0.011798001930183783, -0.8201446642457809], 
'translation': [411.3039349319818, 1180.8903791765097, 0.0]}
'''
ego_to_global_matrix = get_matrix(ego_pose_data0)
print(f"行号[{inspect.currentframe().f_lineno}]:\n",ego_to_global_matrix)
'''
[[-3.45552926e-01  9.38257989e-01  1.62825160e-02  4.11303935e+02]
 [-9.38338111e-01 -3.45280305e-01 -1.74097708e-02  1.18089038e+03]
 [-1.07128245e-02 -2.12945025e-02  9.99715849e-01  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
'''
# 6.3 组装一下矩阵变换，得到lidar_to_global_matrix。 
#   至此。通过这个矩阵，能完成lidar_points -> ego_pose0 -> global的变换
lidar_to_global_matrix = ego_to_global_matrix @ lidar_to_ego_matrix

# 8 。把点进行变换
# 8.0 得到[34688, 4]的数据。 
    # 即 x,y,z -> x,y,z,1
    # 先得到lidar_point_data[:3]  得到xyz 形状是[34688, 3] 
    # 然后统一添加一列1.得到数据形状 [34688, 4]  加一参考变换矩阵乘法
lidar_point = np.concatenate((lidar_point_cloud_data[:, :3], np.ones((len(lidar_point_cloud_data), 1))), axis=1)
# 8.1 
# global_points = lidar_to_global_matrix @ lidar_point_data.T
print(f"行号[{inspect.currentframe().f_lineno}]:\n", lidar_to_global_matrix.shape) #  (4, 4)
print(f"行号[{inspect.currentframe().f_lineno}]:\n", lidar_point.shape) #  (34688, 4)
global_points = lidar_point @ lidar_to_global_matrix.T 
# 等同于上面。相当于求(lidar_to_global_matrix @ lidar_point_data.T).T

# 7  处理camera的数据。处理camera所拍摄的时刻 timestamp = t1时
cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
for cam in cameras:
    # 7.1 得到camera_to_ego_matrix   ego_to_global_matrix  步骤同上
    camera_token = sample["data"][cam]
    camera_data  = nuscenes.get("sample_data", camera_token)
    
    image_file = os.path.join(dataroot, camera_data["filename"])
    image = cv2.imread(image_file)

    camera_calibrated_token = camera_data["calibrated_sensor_token"]
    camera_calibrated_data = nuscenes.get("calibrated_sensor", camera_calibrated_token)
    '''camera多了camera_intrinsic 所以写下来。
    {'token': '75ad8e2a8a3f4594a13db2398430d097', 
    'sensor_token': 'ec4b5d41840a509984f7ec36419d4c09', 
    'translation': [1.52387798135, 0.494631336551, 1.50932822144], 
    'rotation': [0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068], 
    'camera_intrinsic': [[1272.5979470598488, 0.0, 826.6154927353808], [0.0, 1272.5979470598488, 479.75165386361925], [0.0, 0.0, 1.0]]}
    '''
    ego_to_camera_matrix = get_matrix(camera_calibrated_data, True) # 不加True得到的是camera_to_ego_matrix
    camera_ego_pose = nuscenes.get("ego_pose", camera_data["ego_pose_token"])
    global_to_ego_matrix = get_matrix(camera_ego_pose, True) # 不加True得到的是ego_to_global_matrix
    
    ## 7.2 新增步骤，处理camera_intrinsic
    camera_intrinsic = np.eye(4)
    camera_intrinsic[:3, :3] = camera_calibrated_data["camera_intrinsic"] # shape= 【3 * 3】
    ## 7.3 组合两个矩阵  至此。7.3 与 6.3得到的三个矩阵能够完成
    #   timestamp = t0 时的lidar_points -> ego_pose0 -> global  -> (timestamp = t1时)ego_pose1 -> camera -> intrinsic的过程
    global_to_image = camera_intrinsic @ ego_to_camera_matrix @ global_to_ego_matrix
    
    ## 10.2 画框
    ### anns 数据。不同视角用的都是同一份。只不过投影到image上。在camera视锥内的会显示。不在的不显示
    ### anns 数据默认就是global数据
    for token in sample["anns"]:
        annotation = nuscenes.get("sample_annotation", token)
        '''
        {'token': 'ef63a697930c4b20a6b9791f423351da', 
        'sample_token': 'ca9a282c9e77460f8360f564131a8af5', 
        'instance_token': '6dd2cbf4c24b4caeb625035869bca7b5', 
        'visibility_token': '1', 
        'attribute_tokens': ['4d8821270b4a47e3a8a300cbec48188e'], 
        'translation': [373.256, 1130.419, 0.8], 
        'size': [0.621, 0.669, 1.642], 
        'rotation': [0.9831098797903927, 0.0, 0.0, -0.18301629506281616], 
        'prev': '', 
        'next': '7987617983634b119e383d8a29607fd7', 
        'num_lidar_pts': 1,
        'num_radar_pts': 0, 
        'category_name': 'human.pedestrian.adult'}
        '''
        box = Box(annotation["translation"], annotation['size'], Quaternion(annotation["rotation"]))
        # label: nan, score: nan, 
        # xyz: [373.26, 1130.42, 0.80], 
        # wlh: [0.62, 0.67, 1.64], 
        # rot axis: [0.00, 0.00, -1.00], 
        # ang(degrees): 21.09, 
        # ang(rad): 0.37, 
        # vel: nan, nan, nan, 
        # name: None, 
        # token: None
        corners = box.corners().T # box.corners()形状是[3, 8]
        '''
        [[ 3.73679825e+02  1.13058833e+03  1.62100000e+00]
        [ 3.73456358e+02  1.13000893e+03  1.62100000e+00]
        [ 3.73456358e+02  1.13000893e+03 -2.10000000e-02]
        [ 3.73679825e+02  1.13058833e+03 -2.10000000e-02]
        [ 3.73055642e+02  1.13082907e+03  1.62100000e+00]
        [ 3.72832175e+02  1.13024967e+03  1.62100000e+00]
        [ 3.72832175e+02  1.13024967e+03 -2.10000000e-02]
        [ 3.73055642e+02  1.13082907e+03 -2.10000000e-02]]
        '''
        global_corners = np.concatenate((corners, np.ones((len(corners), 1))), axis=1)
        image_base_corners = global_corners @ global_to_image.T # 本来应该是image_base_corners.T = global_to_image @ global_corners.T
        
        image_base_corners[:, :2] /= image_base_corners[:, [2]]
        image_base_corners = image_base_corners.astype(np.int32)
        
        ix, iy = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 1, 2, 3, 0, 5, 6, 7, 4]
        for p0, p1 in zip(image_base_corners[ix], image_base_corners[iy]):
            if p0[2] <= 0 or p1[2] <= 0: continue
            cv2.line(image, (p0[0], p0[1]), (p1[0], p1[1]), (0, 255, 0), 2, 16)
        '''   
        循环表示，
        画0 与 4 的边   画1 与 5 的边   画2 与 6 的边 画3 与 7 的边 
        画0 与 1 的边   画1 与 2 的边   画2 与 3 的边 画3 与 0 的边 
        画4 与 5 的边   画5 与 6 的边   画6 与 7 的边 画7 与 4 的边 
            0 ------ 1
          / |     /  |
        4 ------ 5   |
        |   3 ---|-- 2    
        |  /     | /
        7 ------ 6
        '''
        
    ##8.2 global -> image
    image_points = global_points @ global_to_image.T
    
    # 9.0 转换到iamge上后的数据处理
    ## 9.1 疑问处：缩放关系是否指的是干掉z值。具体指的是什么。为何除以z才能正确显示？
    image_points[:, :2] /= image_points[:, [2]]
    '''
    将点云数据投影到图像上时，通常会使用相机内参和外参来计算投影矩阵。
    在这个过程中，将三维空间中的点云坐标系中的 xy 坐标转换为图像坐标系的过程需要除以 z 分量。
    这是因为相机在拍摄三维场景时，会将三维点投影到二维图像平面上，从而形成二维点云。
    由于相机是中心透视投影，点云数据在图像平面上的坐标与摄像机位置和姿态有关。
    在计算投影矩阵时，需要考虑到相机的内参（如焦距、主点等）和外参（如相机的位置和姿态），
    从而确定点云数据在图像平面上的准确位置。位置。除以 z 分量的过程是一种归一化处理，
    可以消除尺度变化和旋转对投影的影响，从而使投影更加准确。
    '''
    
    ## 9.2 过滤掉z < 0的点
    ## z的中心0点是否就是该相机处。 z<0表示点在图像平面后面，形不成投影。
    for x, y in image_points[image_points[:, 2] > 0, :2].astype(int):
        ## 9.3 之所以没有过滤x y 是因为circle对于越界的数值。自动不绘制。
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1, 16)
        
    cv2.imwrite(f"{cam}.jpg", image)
    
    
    