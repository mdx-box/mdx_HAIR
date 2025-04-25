#将所有的RGB和skeleton加载为npy文件，便于后续的使用
import numpy as np
from utils.tool import load_config, str_to_float
import pandas as pd
import os
from PIL import Image


def skeleton_every_frame(path:str):
    """
    :param path: 骨骼数据集地址，来自json配置文件
    """
    skeleton_frame = pd.read_csv(path, sep='\t', header=None)
    #读入数据，并转换为list
    X = str_to_float(skeleton_frame[3][1:].tolist())
    Y = str_to_float(skeleton_frame[4][1:].tolist())
    Z = str_to_float(skeleton_frame[5][1:].tolist())
    Qw = str_to_float(skeleton_frame[6][1:].tolist())
    Qx = str_to_float(skeleton_frame[7][1:].tolist())
    Qy = str_to_float(skeleton_frame[8][1:].tolist())
    Qz = str_to_float(skeleton_frame[9][1:].tolist())

    return np.stack((X,Y,Z,Qw,Qx,Qy,Qz),axis=1)
def load_all_skeleton_to_npy_every_video(path:str):
    """
    :param path: skeleton数据上级目录
    """
    all_skeleton_data = []
    skeleton_path_list = os.listdir(path)
    for skeleton_path in skeleton_path_list:
        single_skeleton = skeleton_every_frame(path=f'{path}/{skeleton_path}')
        all_skeleton_data.append(single_skeleton)

    return np.stack(all_skeleton_data,axis=0)   #最终输出维度:(2921,32,7)

def RGB_every_frame(path:str):
    """
    :param path: RGB数据集地址，加载单张RGB图片
    """
    single_RGB = np.array(Image.open(path))

    return single_RGB
def load_all_RGB_to_npy_every_video(path):
    """
    :param path: 将单个视频中的所有帧全部存入npy文件
    """
    RGB_all_list = []
    RGB_path_list = os.listdir(path)
    for RGB_path in RGB_path_list:
        RGB_all_list.append(RGB_every_frame(path=f'{path}/{RGB_path}'))
    return np.stack(RGB_all_list,axis=0)

def label_to_npy(path:str):
    """
    :param path将读入的label值存入npy文件中
    """
    label_path = load_config('label_path')
    label_frame = pd.read_csv(label_path,sep=' ',header=None)
    label_list = label_frame[1].tolist()
    return np.array(label_list)






