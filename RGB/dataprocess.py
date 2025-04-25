#处理RGB输入
import os
import numpy as np
from utils.tool import load_config
from PIL import Image
from tqdm import tqdm
import cv2 as cv
import pandas as pd
def train_img_and_label_data_format(root_path_image:str,zero_index:list, label_frame:pd.DataFrame):
    #这里就是将每一个文件中的视频转换为以32帧为单位数据，并将Others动作去除
    image_list = os.listdir(root_path_image)
    all_image = []
    all_label = []
    stage_image = []
    stage_label = []
    count = 0
    for i in range(len(image_list)):
        if i in zero_index:
            continue
        else:
            img = cv.imread(filename=f'{root_path_image}/{image_list[i]}')
            stage_image.append(img)
            read_label = label_frame.loc[i].tolist()
            stage_label.append(read_label)
            count += 1
            if count % 32 == 0:
                all_image.append(np.stack(stage_image, axis=0))
                all_label.append(np.stack(stage_label,axis=0))
                stage_image = []
                stage_label = []
    img_out = np.stack(all_image, axis=0)
    label_out = np.stack(all_label, axis=0)
    return img_out, label_out

if __name__ == '__main__':
    #开始测试
    root_labels = 'E:/mdxDataset'
    file_path = os.listdir(root_labels)
    for path in tqdm(file_path, desc="All dataset processing ..."):
        every_label_file = f'{root_labels}/{path}/{path}/Labels.txt'
        labels_frame = pd.read_csv(every_label_file, sep=" ", header=None)
        zero_index = zero_index=[index for index, x in enumerate(labels_frame[1].tolist()) if x == 0]
        df = pd.get_dummies(labels_frame[labels_frame[1]>0][1],dtype=int)
        img_out, label_out = train_img_and_label_data_format(root_path_image=f'resize_img/{path}',zero_index=zero_index, label_frame=df)
        os.mkdir(path=f'Input_dataset/{path}')
        np.save(f'Input_dataset/{path}/{path}_RGB.npy',img_out)
        np.save(f'Input_dataset/{path}/{path}_labels.npy',label_out)
        print(f'RGB and corresponding labels have been saved, shape of RGB is {img_out.shape}, shape of labels is {label_out.shape}')
    
