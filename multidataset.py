import sys
sys.path.append('D:/mdx/HAIR_R.1/HAIR')
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class SkeletonDataset(Dataset):
    def __init__(self, data_dir, joint_mask_rate=0.0, frame_mask_rate=0.0):
        """
        初始化数据集
        Args:
            data_dir: 包含数据的目录路径，应该包含多个.npz文件
            joint_mask_rate: 关节点mask比率，范围[0, 1]
            frame_mask_rate: 帧级mask比率，范围[0, 1]
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if not self.data_files:
            raise ValueError(f"数据目录中没有.npz文件: {data_dir}")
        
        # 预先加载所有数据
        self.all_data = []
        self.all_labels = []
        
        for file in self.data_files:
            file_path = os.path.join(data_dir, file)
            try:
                data_dict = np.load(file_path)
                if 'data' not in data_dict or 'label' not in data_dict:
                    raise ValueError(f"文件{file}中缺少'data'或'label'字段")
                self.all_data.append(data_dict['data'])
                self.all_labels.append(data_dict['label'])
            except Exception as e:
                raise ValueError(f"加载文件{file}时出错: {str(e)}")
        
        if not self.all_data:
            raise ValueError(f"没有成功加载任何数据: {data_dir}")
            
        # 将所有数据拼接成一个大数组
        try:
            self.data = np.concatenate(self.all_data, axis=0)
            self.labels = np.concatenate(self.all_labels, axis=0)
        except Exception as e:
            raise ValueError(f"拼接数据时出错，请检查数据格式是否一致: {str(e)}")
            
        # 设置mask rate属性
        self.joint_mask_rate = joint_mask_rate
        self.frame_mask_rate = frame_mask_rate
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取一个数据样本
        Args:
            idx: 样本索引
        Returns:
            data: 形状为(C, T, V, M)的骨骼数据，其中：
                C: 通道数
                T: 帧数
                V: 关节点数
                M: person数（在这里固定为1）
            label: one-hot编码的标签
        """
        data = self.data[idx]  # (C, T, V, M)
        label = self.labels[idx]  # (num_classes,)
        
        # 转换为float32类型
        data = data.astype(np.float32)
        label = label.astype(np.float32)
        
        # 应用mask操作
        if self.joint_mask_rate > 0:
            data = self.joint_mask(data)
        if self.frame_mask_rate > 0:
            data = self.frame_mask(data)
        
        return data, label
    
    def joint_mask(self, data):
        """
        对骨骼数据进行关节点mask
        Args:
            data: shape为(C, T, V, M)的骨骼数据
        Returns:
            masked_data: 被mask后的数据
        """
        if not 0 <= self.joint_mask_rate <= 1:
            raise ValueError("joint_mask_rate必须在0到1之间")
            
        masked_data = data.copy()
        C, T, V = data.shape
        
        # 对每一帧进行mask
        for t in range(T):
            # 随机选择要mask的关节点
            mask_indices = np.random.choice(
                V,
                size=int(V * self.joint_mask_rate),
                replace=False
            )
            # 将选中的关节点坐标设为0
            masked_data[:, t, mask_indices, :] = 0
            
        return masked_data
    
    def frame_mask(self, data):
        """
        对骨骼数据进行帧级mask
        Args:
            data: shape为(C, T, V, M)的骨骼数据
        Returns:
            masked_data: 被mask后的数据
        """
        if not 0 <= self.frame_mask_rate <= 1:
            raise ValueError("frame_mask_rate必须在0到1之间")
            
        masked_data = data.copy()
        C, T, V, M = data.shape
        
        # 随机选择要mask的帧
        mask_frames = np.random.choice(
            T,
            size=int(T * self.frame_mask_rate),
            replace=False
        )
        # 将选中的帧的所有关节点坐标设为0
        masked_data[:, mask_frames, :, :] = 0
        
        return masked_data





class RGBDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
            
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if not self.data_files:
            raise ValueError(f"数据目录中没有.npz文件: {data_dir}")
            
        self.data_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字顺序排序
        
        # 预加载所有数据
        self.all_data = []
        self.all_labels = []
        for file in self.data_files:
            try:
                data = np.load(os.path.join(data_dir, file))
                if 'data' not in data or 'label' not in data:
                    raise ValueError(f"文件{file}中缺少'data'或'label'字段")
                self.all_data.append(data['data'])
                self.all_labels.append(data['label'])
            except Exception as e:
                raise ValueError(f"加载文件{file}时出错: {str(e)}")
        
        if not self.all_data:
            raise ValueError(f"没有成功加载任何数据: {data_dir}")
            
        try:
            self.all_data = np.concatenate(self.all_data, axis=0)
            self.all_labels = np.concatenate(self.all_labels, axis=0)
        except Exception as e:
            raise ValueError(f"拼接数据时出错，请检查数据格式是否一致: {str(e)}")
        
        # 随机选择所有数据作为训练集
        np.random.seed(42)  # 设置随机种子以确保可重复性
        total_samples = len(self.all_data)
        train_size = int(total_samples * 1)
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        
        self.all_data = self.all_data[train_indices]
        self.all_labels = self.all_labels[train_indices]
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.all_data[idx]).float()
        label = torch.tensor(self.all_labels[idx]).long()
        return data, label


class MultiModalDataset(Dataset):
    def __init__(self, data_dir="D:/mdx/HAIR_R.1/dataset", joint_mask_rate=0.0, frame_mask_rate=0.0):
        """初始化多模态数据集
        Args:
            data_dir: 数据根目录路径，默认为'D:/mdx/HAIR_R.1/dataset'
            joint_mask_rate: 关节点mask比率，范围[0, 1]
            frame_mask_rate: 帧级mask比率，范围[0, 1]
        """
        rgb_data_dir = f'{data_dir}/RGB'
        skeleton_data_dir = f'{data_dir}/skeleton'
        
        # 检查数据目录是否存在
        if not os.path.exists(rgb_data_dir):
            raise FileNotFoundError(f"RGB数据目录不存在: {rgb_data_dir}")
        if not os.path.exists(skeleton_data_dir):
            raise FileNotFoundError(f"骨骼数据目录不存在: {skeleton_data_dir}")
            
        # 检查数据文件是否存在
        rgb_files = sorted([f for f in os.listdir(rgb_data_dir) if f.endswith('.npz')], key=lambda x: int(x.split('.')[0]))
        skeleton_files = sorted([f for f in os.listdir(skeleton_data_dir) if f.endswith('.npz')], key=lambda x: int(x.split('.')[0]))
        
        if not rgb_files:
            raise ValueError(f"RGB数据目录中没有.npz文件: {rgb_data_dir}")
        if not skeleton_files:
            raise ValueError(f"骨骼数据目录中没有.npz文件: {skeleton_data_dir}")
        
        # 确保文件列表一致
        if rgb_files != skeleton_files:
            raise ValueError("RGB数据和骨骼数据的文件列表不一致")
            
        # 预加载所有数据
        self.rgb_data = []
        self.skeleton_data = []
        self.labels = []
        
        for rgb_file, skeleton_file in zip(rgb_files, skeleton_files):
            rgb_path = os.path.join(rgb_data_dir, rgb_file)
            skeleton_path = os.path.join(skeleton_data_dir, skeleton_file)
            
            rgb_data = np.load(rgb_path)
            skeleton_data = np.load(skeleton_path)
            
            self.rgb_data.append(rgb_data['data'])
            self.skeleton_data.append(skeleton_data['data'])
            self.labels.append(skeleton_data['label'])  # 使用骨骼数据的标签
        
        # 将数据拼接成大数组
        self.rgb_data = np.concatenate(self.rgb_data, axis=0)
        self.skeleton_data = np.concatenate(self.skeleton_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        # 设置mask rate属性
        self.joint_mask_rate = joint_mask_rate
        self.frame_mask_rate = frame_mask_rate
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        """获取一个多模态数据样本
        Args:
            idx: 样本索引
        Returns:
            rgb_data: RGB数据
            skeleton_data: 骨骼数据
            label: 标签
        """
        rgb_data = torch.from_numpy(self.rgb_data[idx]).float()
        skeleton_data = self.skeleton_data[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # 应用mask操作到骨骼数据
        if self.joint_mask_rate > 0:
            skeleton_data = self.joint_mask(skeleton_data)
        if self.frame_mask_rate > 0:
            skeleton_data = self.frame_mask(skeleton_data)
        
        skeleton_data = torch.from_numpy(skeleton_data).float()
        label = torch.from_numpy(label).long()
        
        return rgb_data, skeleton_data, label
        
    def joint_mask(self, data):
        """对骨骼数据进行关节点mask"""
        if not 0 <= self.joint_mask_rate <= 1:
            raise ValueError("joint_mask_rate必须在0到1之间")
            
        masked_data = data.copy()
        C, T, V, M = data.shape
        
        # 对每一帧进行mask
        for t in range(T):
            mask_indices = np.random.choice(
                V,
                size=int(V * self.joint_mask_rate),
                replace=False
            )
            masked_data[:, t, mask_indices, :] = 0
            
        return masked_data
    
    def frame_mask(self, data):
        """对骨骼数据进行帧级mask"""
        if not 0 <= self.frame_mask_rate <= 1:
            raise ValueError("frame_mask_rate必须在0到1之间")
            
        masked_data = data.copy()
        C, T, V, M = data.shape
        
        mask_frames = np.random.choice(
            T,
            size=int(T * self.frame_mask_rate),
            replace=False
        )
        masked_data[:, mask_frames, :, :] = 0
        
        return masked_data




def main():
    # 测试SkeletonDataset的DataLoader
    print("\n测试SkeletonDataset的DataLoader:")
    skeleton_dataset = SkeletonDataset("d:/mdx/HAIR_R.1/dataset/skeleton", joint_mask_rate=0.2, frame_mask_rate=0.1)
    skeleton_loader = DataLoader(dataset=skeleton_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    batch_data, batch_label = next(iter(skeleton_loader))
    print(f"骨骼数据batch shape: {batch_data.shape}")
    print(f"标签batch shape: {batch_label.shape}")
    print(f"数据集大小: {len(skeleton_dataset)}")
    
    # 测试RGBDataset的DataLoader
    print("\n测试RGBDataset的DataLoader:")
    rgb_dataset = RGBDataset("d:/mdx/HAIR_R.1/dataset/RGB")
    rgb_loader = DataLoader(dataset=rgb_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    batch_data, batch_label = next(iter(rgb_loader))
    print(f"RGB数据batch shape: {batch_data.shape}")
    print(f"标签batch shape: {batch_label.shape}")
    print(f"数据集大小: {len(rgb_dataset)}")
    
    # 测试MultiModalDataset的DataLoader
    print("\n测试MultiModalDataset的DataLoader:")
    multi_dataset = MultiModalDataset("d:/mdx/HAIR_R.1/dataset", joint_mask_rate=0.2, frame_mask_rate=0.1)
    multi_loader = DataLoader(dataset=multi_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    batch_rgb, batch_skeleton, batch_label = next(iter(multi_loader))
    print(f"RGB数据batch shape: {batch_rgb.shape}")
    print(f"骨骼数据batch shape: {batch_skeleton.shape}")
    print(f"标签batch shape: {batch_label.shape}")
    print(f"数据集大小: {len(multi_dataset)}")

if __name__ == '__main__':
    main()