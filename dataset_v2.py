from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
class MyRGBSkeletonDataset(Dataset):
    def __init__(self, data_root_path:str, joint_mask_rate=0.1, frame_mask_rate=0.1, brightness_factor=1.1, train=True, seed=42):
        self.brightness_factor = brightness_factor
        torch.manual_seed(seed)  # 设置随机种子以确保可复现性
        super(MyRGBSkeletonDataset, self).__init__()
        self.data_root_path = data_root_path
        self.joint_mask_rate = joint_mask_rate
        self.frame_mask_rate = frame_mask_rate
        full_rgb_data, full_skeleton_data, full_labels = self.data_load(data_path=self.data_root_path)
        
        # 划分训练集和测试集
        train_size = int(1 * len(full_labels))  # 71%用于训练
        test_size = len(full_labels) - train_size   # 29%用于测试
        
        indices = torch.randperm(len(full_labels))
        if train:
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
            
        self.RGB_data = full_rgb_data
        self.Skeleton = full_skeleton_data
        self.labels = full_labels
        self.len = len(self.labels)        
        print('数据已经准备好了....')
    def __getitem__(self, index) :
        rgb_data = self.RGB_data[index].clone()
        skeleton_data = self.Skeleton[index].clone()
        
        # 应用joint mask和frame mask
        if self.joint_mask_rate > 0:
            skeleton_data = self.joint_mask(skeleton_data)
        if self.frame_mask_rate > 0:
            skeleton_data = self.frame_mask(skeleton_data)
            
        # 调整RGB数据的亮度
        if self.brightness_factor != 1.0:
            # 对每一帧分别进行亮度调整
            T, H, W, C = rgb_data.shape
            adjusted_frames = []
            for t in range(T):
                frame = rgb_data[t]
                # print(frame.shape) torch.Size([112, 112, 3])
                adjusted_frame = TF.adjust_brightness(frame.permute(2,0,1), self.brightness_factor) # (3,112,112)
                # 可视化第一帧的对比图
                # if t == 0:
                #     self.visualize_brightness_adjustment(frame, adjusted_frame)
                adjusted_frames.append(adjusted_frame.permute(1,2,0))  # (112,112,3)
            rgb_data = torch.stack(adjusted_frames, dim=0)
            
        return rgb_data, skeleton_data, self.labels[index]

    def joint_mask(self, data):
        """对骨骼数据进行关节点mask
        Args:
            data: shape为(T=16, C=3, V=33)的骨骼数据，其中T是帧数，C是xyz坐标，V是骨骼点数
        Returns:
            masked_data: 被mask后的数据
        """
        if not 0 <= self.joint_mask_rate <= 1:
            raise ValueError("joint_mask_rate必须在0到1之间")
            
        masked_data = data.clone()
        T, C, V = data.shape
        
        # 随机选择要mask的关节点
        mask_joints = torch.randperm(V)[:int(V * self.joint_mask_rate)]
        
        # 将选中的关节点的所有时间帧的坐标设为0
        masked_data[:, :, mask_joints] = 0
        
        return masked_data
        
    def frame_mask(self, data):
        """对骨骼数据进行帧级mask
        Args:
            data: shape为(T=16, C=3, V=33)的骨骼数据，其中T是帧数，C是xyz坐标，V是骨骼点数
        Returns:
            masked_data: 被mask后的数据
        """
        if not 0 <= self.frame_mask_rate <= 1:
            raise ValueError("frame_mask_rate必须在0到1之间")
            
        masked_data = data.clone()
        T, C, V = data.shape
        
        # 随机选择要mask的帧
        mask_frames = torch.randperm(T)[:int(T * self.frame_mask_rate)]
        
        # 将选中的帧的所有关节点坐标设为0
        masked_data[mask_frames, :, :] = 0
        
        return masked_data
    def __len__(self):
        return self.len

    def visualize_brightness_adjustment(self, frame, adjusted_frame):
        """可视化原始帧和调整亮度后的帧
        Args:
            frame: 原始帧数据
            adjusted_frame: 调整亮度后的帧数据
        """
        plt.figure(figsize=(10, 5))
        
        # 显示原始帧
        plt.subplot(121)
        plt.title('原始帧')
        plt.imshow(frame)
        plt.axis('off')
        
        # 显示调整亮度后的帧
        plt.subplot(122)
        plt.title(f'调整亮度后的帧 (factor={self.brightness_factor})')
        plt.imshow(adjusted_frame.permute(1, 2, 0))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

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
    


# if __name__ == '__main__':
#     # 创建不同亮度的数据集实例
#     dataset_normal = MyRGBSkeletonDataset(data_root_path='D:/mdx/HAIR_R.1/dataset', 
#                                         joint_mask_rate=0, 
#                                         frame_mask_rate=0, 
#                                         brightness_factor=1.0)
#     dataset_bright = MyRGBSkeletonDataset(data_root_path='D:/mdx/HAIR_R.1/dataset', 
#                                          joint_mask_rate=0, 
#                                          frame_mask_rate=0, 
#                                          brightness_factor=1.5)
#     dataset_dark = MyRGBSkeletonDataset(data_root_path='D:/mdx/HAIR_R.1/dataset', 
#                                        joint_mask_rate=0, 
#                                        frame_mask_rate=0, 
#                                        brightness_factor=0.5)
    
#     print(f'数据集大小: {len(dataset_normal)}')
    
#     # 创建DataLoader
#     dataloader_normal = DataLoader(dataset=dataset_normal, batch_size=1, shuffle=True, num_workers=0)
#     dataloader_bright = DataLoader(dataset=dataset_bright, batch_size=1, shuffle=True, num_workers=0)
#     dataloader_dark = DataLoader(dataset=dataset_dark, batch_size=1, shuffle=True, num_workers=0)
    
#     # 获取一个样本进行可视化
#     rgb_normal, _, _ = next(iter(dataloader_normal))
#     rgb_bright, _, _ = next(iter(dataloader_bright))
#     rgb_dark, _, _ = next(iter(dataloader_dark))
#     # print(rgb_normal[0, 0].shape)
#     # 创建图像网格进行对比
#     plt.figure(figsize=(15, 5))
    
#     # 显示原始亮度图像
#     plt.subplot(131)
#     plt.title('Normal Brightness (factor=1.0)')
#     plt.imshow(rgb_normal[0, 0])
#     plt.axis('off')
    
#     # 显示增加亮度后的图像
#     plt.subplot(132)
#     plt.title('Increased Brightness (factor=1.5)')
#     plt.imshow(rgb_bright[0, 0].permute(2, 1, 0))
#     plt.axis('off')
    
#     # 显示降低亮度后的图像
#     plt.subplot(133)
#     plt.title('Decreased Brightness (factor=0.5)')
#     plt.imshow(rgb_dark[0, 0].permute(2, 2, 0))
#     plt.axis('off')
    
#     plt.tight_layout()
#     plt.show()