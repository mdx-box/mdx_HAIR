import torch
from torch.utils.data import DataLoader
from dataset_v2 import MyRGBSkeletonDataset
import os
from tqdm import tqdm
import numpy as np
import time
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix

def load_model(model_path, device='cuda'):
    """加载预训练模型
    Args:
        model_path: 模型权重文件路径
        device: 运行设备，默认为cuda
    Returns:
        model: 加载了权重的模型
    """
    if os.path.exists(model_path):
        try:
            model = torch.load(model_path, map_location=device)
            print(f'成功加载整个模型: {model_path}')
        except:
            model = MultiviewMultimodalModel()  # Initialize model
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'成功加载模型权重: {model_path}')
    else:
        raise FileNotFoundError(f'模型文件不存在: {model_path}')
    
    model = model.to(device)
    model.eval()
    return model

def inference_with_progressive_frames(model, dataloader, device='cuda'):
    """执行逐帧推理并记录每帧的预测结果
    Args:
        model: 预训练模型
        dataloader: 测试数据加载器
        device: 运行设备
    Returns:
        frame_predictions: 每帧的预测结果
        labels: 真实标签
        frame_accuracies: 每帧的准确率
        inference_stats: 推理统计信息
    """
    frame_predictions = {i: [] for i in range(1, 17)}  # 存储每帧的预测结果
    frame_confidences = {i: [] for i in range(1, 17)}  # 存储每帧的置信度
    labels = []
    frame_accuracies = {i: [] for i in range(1, 17)}  # 存储每帧的准确率
    inference_stats = {
        'per_video_time': [], 
        'total_time': 0,
        'video_paths': [],
        'video_times': {},
        'frame_accuracies': {},
        'confusion_matrices': {},
        'per_class_predictions': {}
    }
    
    total_start_time = time.time()
    
    with torch.no_grad():
        for rgb_data, skeleton_data, batch_labels in tqdm(dataloader, desc="Progressive Frame Inference"):
            batch_start_time = time.time()
            batch_size = rgb_data.size(0)
            
            # 记录视频信息
            start_idx = len(inference_stats['per_video_time'])
            video_paths = [f"video_{start_idx + i + 1}" for i in range(batch_size)]
            inference_stats['video_paths'].extend(video_paths)
            
            # 逐帧处理
            for frame_count in range(1, 17):
                # 创建填充后的数据
                padded_rgb = rgb_data.clone()
                padded_skeleton = skeleton_data.clone()
                
                # 将未使用的帧填充为0
                if frame_count < 16:
                    padded_rgb[:, frame_count:, :, :, :] = 0
                    padded_skeleton[:, frame_count:, :, :] = 0
                
                # 准备数据
                padded_rgb = padded_rgb.to(torch.float)
                padded_skeleton = padded_skeleton.to(torch.float)
                padded_rgb = rearrange(padded_rgb, 'b t h w c -> b c t h w').to(device)
                padded_skeleton = padded_skeleton.to(device)
                batch_labels = batch_labels.to(device)
                
                # 模型推理
                rgb_inputs = [padded_rgb.clone() for _ in range(3)]
                outputs = model(rgb_inputs, padded_skeleton)
                
                # 获取预测结果和置信度
                confidences, predictions = torch.max(outputs.softmax(dim=1), dim=1)
                
                if len(batch_labels.shape) > 1:
                    batch_labels = torch.argmax(batch_labels, dim=1)
                
                # 记录结果
                frame_predictions[frame_count].extend(predictions.cpu().numpy())
                frame_confidences[frame_count].extend(confidences.cpu().numpy())
                if frame_count == 1:  # 只需要记录一次标签
                    labels.extend(batch_labels.cpu().numpy())
                
                # 计算准确率
                correct = (predictions == batch_labels).sum().item()
                accuracy = correct / batch_size
                frame_accuracies[frame_count].append(accuracy)
            
            # 记录处理时间
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            per_video_time = batch_time / batch_size
            
            for video_path in video_paths:
                inference_stats['video_times'][video_path] = per_video_time
                inference_stats['per_video_time'].append(per_video_time)
    
    # 计算总体统计信息
    inference_stats['total_time'] = time.time() - total_start_time
    
    # 计算每帧的平均准确率和每个类别的预测结果
    unique_labels = np.unique(labels)
    for frame_count in range(1, 17):
        inference_stats['frame_accuracies'][str(frame_count)] = np.mean(frame_accuracies[frame_count])
        
        # 计算每帧的混淆矩阵
        if len(labels) > 0:
            cm = confusion_matrix(labels, frame_predictions[frame_count])
            inference_stats['confusion_matrices'][str(frame_count)] = cm.tolist()
            
            # 计算每个类别的预测结果
            frame_preds = np.array(frame_predictions[frame_count])
            frame_confs = np.array(frame_confidences[frame_count])
            frame_key = str(frame_count)
            
            if frame_key not in inference_stats['per_class_predictions']:
                inference_stats['per_class_predictions'][frame_key] = {}
            
            for label in unique_labels:
                label_mask = (np.array(labels) == label)
                class_preds = frame_preds[label_mask]
                class_confs = frame_confs[label_mask]
                
                class_stats = {
                    'total_samples': int(np.sum(label_mask)),
                    'correct_predictions': int(np.sum(class_preds == label)),
                    'predictions': class_preds.tolist(),
                    'confidences': class_confs.tolist(),
                    'accuracy': float(np.mean(class_preds == label))
                }
                
                inference_stats['per_class_predictions'][frame_key][str(int(label))] = class_stats
    
    # 保存统计信息
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stats_file = f"progressive_inference_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(inference_stats, f, indent=4)
    print(f"\n推理统计数据已保存到: {stats_file}")
    
    return frame_predictions, labels, frame_accuracies, inference_stats

def visualize_progressive_results(frame_accuracies, inference_stats, dataset=None):
    """可视化逐帧预测结果"""
    # 绘制准确率随帧数的变化
    plt.figure(figsize=(10, 6))
    frame_nums = list(inference_stats['frame_accuracies'].keys())
    accuracies = list(inference_stats['frame_accuracies'].values())
    
    plt.plot(frame_nums, accuracies, marker='o')
    plt.title('Acccuray with frames')
    plt.xlabel('Frames')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    # 为最后一帧绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    final_cm = np.array(inference_stats['confusion_matrices']['16'])
    
    # 获取类别名称
    try:
        class_names = dataset.class_names if dataset else None
    except AttributeError:
        class_names = [str(i) for i in range(final_cm.shape[0])]
    
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('最终帧的混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def main():
    """主函数，执行模型推理和结果可视化"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据集参数
    data_root = 'D:/mdx/HAIR_R.1/dataset'
    joint_mask_rate = 0.3
    frame_mask_rate = 0.2
    brightness_factor = 1.0
    
    # 加载测试数据集
    test_dataset = MyRGBSkeletonDataset(
        data_root_path=data_root,
        joint_mask_rate=joint_mask_rate,
        frame_mask_rate=frame_mask_rate,
        brightness_factor=brightness_factor,
        train=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    # 加载模型
    model_path = 'D:/mdx/HAIR_R.1/HAIR/pth/model_epoch_142_acc_0.9583_44.pth'
    model = load_model(model_path, device)
    
    # 执行逐帧推理
    frame_predictions, labels, frame_accuracies, inference_stats = \
        inference_with_progressive_frames(model, test_loader, device)
    
    # 可视化结果
    visualize_progressive_results(frame_accuracies, inference_stats, test_dataset)
    
    # 输出每帧的准确率
    print("\n每帧的准确率:")
    for frame_num, acc in inference_stats['frame_accuracies'].items():
        print(f"帧 {frame_num}: {acc:.4f}")

if __name__ == '__main__':
    main()