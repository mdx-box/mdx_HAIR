from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm.auto import tqdm
class MyRGBSkeletonDataset(Dataset):
    def __init__(self, data_root_path:str):

        super(MyRGBSkeletonDataset, self).__init__()
        self.data_root_path = data_root_path
        self.RGB_data, self.Skeleton, self.labels = self.data_load(data_path=self.data_root_path)
        self.len = len(self.labels)
        print('数据已经准备好了....')
    def __getitem__(self, index) :

        return self.RGB_data[index], self.Skeleton[index], self.labels[index]
    def __len__(self):
        return self.len

    def data_load(self, data_path):
        # 获取RGB和skeleton数据路径
        rgb_path = os.path.join(data_path, 'RGB')
        skeleton_path = os.path.join(data_path, 'skeleton')
        
        # 确保路径存在
        if not os.path.exists(rgb_path) or not os.path.exists(skeleton_path):
            raise ValueError(f"RGB或skeleton路径不存在: {rgb_path}, {skeleton_path}")
        
        # 获取npz文件列表 (1.npz到9.npz)
        npz_files = [f"{i}.npz" for i in range(1, 10)]
        
        X_RGB = []
        X_Skeleton = []
        y = []
        
        # 加载每个npz文件
        for npz_file in tqdm(npz_files, desc="加载数据"):
            # 加载RGB数据
            rgb_file_path = os.path.join(rgb_path, npz_file)
            if os.path.exists(rgb_file_path):
                rgb_data = np.load(rgb_file_path)
                rgb_array = torch.from_numpy(rgb_data['data'])
                X_RGB.append(rgb_array)
                
                # 获取标签（只从RGB数据中获取一次）
                label_array = torch.from_numpy(rgb_data['label'])
                y.append(label_array)
            else:
                print(f"警告: RGB文件不存在 {rgb_file_path}")
                continue
            
            # 加载对应的skeleton数据
            skeleton_file_path = os.path.join(skeleton_path, npz_file)
            if os.path.exists(skeleton_file_path):
                skeleton_data = np.load(skeleton_file_path)
                skeleton_array = torch.from_numpy(skeleton_data['data'])
                X_Skeleton.append(skeleton_array)
            else:
                print(f"警告: Skeleton文件不存在 {skeleton_file_path}")
                # 如果skeleton文件不存在，移除对应的RGB和标签数据
                X_RGB.pop()
                y.pop()
        
        # 确保有数据被加载
        if not X_RGB or not X_Skeleton or not y:
            raise ValueError("没有数据被加载，请检查数据路径和文件格式")
        
        # 合并所有数据
        return torch.cat(X_RGB, dim=0), torch.cat(X_Skeleton, dim=0), torch.cat(y, dim=0)
    


if __name__ == '__main__':
    dataset = MyRGBSkeletonDataset(data_root_path='D:/mdx/HAIR_R.1/dataset')
    print(f'数据集大小: {len(dataset)}')
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last= True
    )
    
    # 测试数据加载
    for batch_rgb, batch_skeleton, batch_labels in dataloader:
        print(f'RGB数据形状: {batch_rgb.shape}')
        print(f'Skeleton数据形状: {batch_skeleton.shape}')
        print(f'标签形状: {batch_labels.shape}')