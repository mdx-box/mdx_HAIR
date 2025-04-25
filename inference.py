from msilib import PID_REVNUMBER
import torch
from torch.utils.data import DataLoader
from dataset_v2 import MyRGBSkeletonDataset
# from Multiview_Multimodal_model import MultiviewMultimodalModel
import os
from tqdm import tqdm
import numpy as np
import time
from torch.autograd.profiler import profile
from thop import clever_format
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix

# Placeholder for model and dataset (uncomment and adjust as needed)
# class MultiviewMultimodalModel(torch.nn.Module):
#     def __init__(self):
#         super(MultiviewMultimodalModel, self).__init__()
#         # Define your model architecture here
#         self.dummy = torch.nn.Linear(1, 1)  # Placeholder
#
#     def forward(self, rgb_inputs, skeleton_input):
#         # Define forward pass
#         return self.dummy(rgb_inputs[0].mean(dim=(1,2,3,4)))  # Placeholder
#
# class MyRGBSkeletonDataset(torch.utils.data.Dataset):
#     def __init__(self, data_root_path, joint_mask_rate, frame_mask_rate, brightness_factor, train=True):
#         self.data_root_path = data_root_path
#         self.joint_mask_rate = joint_mask_rate
#         self.frame_mask_rate = frame_mask_rate
#         self.brightness_factor = brightness_factor
#         self.train = train
#         # Placeholder data
#         self.RGB_data = [torch.randn(30, 224, 224, 3) for _ in range(100)]  # Dummy RGB data
#         self.skeleton_data = [torch.randn(30, 25, 3) for _ in range(100)]  # Dummy skeleton data
#         self.labels = [torch.randint(0, 5, (1,)).item() for _ in range(100)]  # Dummy labels
#         self.class_names = ['action1', 'action2', 'action3', 'action4', 'action5']  # Placeholder class names
#
#     def __len__(self):
#         return len(self.RGB_data)
#
#     def __getitem__(self, idx):
#         rgb = self.RGB_data[idx]
#         skeleton = self.skeleton_data[idx]
#         label = self.labels[idx]
#         label_one_hot = torch.zeros(len(self.class_names))
#         label_one_hot[label] = 1.0
#         return rgb, skeleton, label_one_hot

def load_model(model_path, device='cuda'):
    """加载预训练模型
    Args:
        model_path: 模型权重文件路径
        device: 运行设备，默认为cuda
    Returns:
        model: 加载了权重的模型
    """
    # Initialize model (uncomment and replace with actual model)
    # model = MultiviewMultimodalModel()
    if os.path.exists(model_path):
        # Adjust based on whether the entire model or state dict was saved
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

def inference(model, dataloader, device='cuda'):
    """执行推理并计算统计信息
    Args:
        model: 预训练模型
        dataloader: 测试数据加载器
        device: 运行设备
    Returns:
        predictions: 预测结果
        labels: 真实标签
        accuracies: 每批准确率
        inference_stats: 推理统计信息
    """
    predictions = []
    labels = []
    accuracies = []
    inference_stats = {
        'per_video_time': [], 
        'total_time': 0, 
        'flops': [],
        'video_paths': [],
        'video_times': {},
        'class_accuracies': {},
        'confusion_matrix': None
    }
    
    total_start_time = time.time()
    
    with torch.no_grad():
        for rgb_data, skeleton_data, batch_labels in tqdm(dataloader, desc="Inference"):
            batch_start_time = time.time()
            rgb_data, skeleton_data = rgb_data.to(torch.float), skeleton_data.to(torch.float)
            rgb_data = rearrange(rgb_data, 'b t h w c -> b c t h w').to(device)
            skeleton_data = skeleton_data.to(device)
            batch_labels = batch_labels.to(device)
            
            rgb_inputs = [rgb_data.clone() for _ in range(3)]
            skeleton_input = skeleton_data.clone()
            print(f'rgb_inputs[0].shape: {rgb_inputs[0].shape}')
            with profile(use_cuda=(device == 'cuda')) as prof:
                outputs = model(rgb_inputs, skeleton_input)
            
            total_flops = sum(int(e.flops) for e in prof.function_events if e.flops > 0)
            inference_stats['flops'].append(total_flops)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            batch_size = rgb_data.size(0)
            start_idx = len(inference_stats['per_video_time'])
            video_paths = [f"video_{start_idx + i + 1}" for i in range(batch_size)]
            inference_stats['video_paths'].extend(video_paths)
            
            per_video_time = batch_time / batch_size
            for video_path in video_paths:
                inference_stats['video_times'][video_path] = per_video_time
                inference_stats['per_video_time'].append(per_video_time)
            
            _, predicted = torch.max(outputs.data, 1)
            
            if len(batch_labels.shape) > 1:
                batch_labels = torch.argmax(batch_labels, dim=1)
            
            correct = (predicted == batch_labels).sum().item()
            accuracy = correct / batch_labels.size(0)
            
            predictions.extend(predicted.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            accuracies.append(accuracy)
    
    inference_stats['total_time'] = time.time() - total_start_time
    avg_flops = np.mean(inference_stats['flops']) if inference_stats['flops'] else 0
    inference_stats['flops'] = clever_format([avg_flops], "%.3f")[0]
    
    # Calculate per-class accuracies
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    for cls in unique_classes:
        cls_mask = (labels == cls)
        cls_count = cls_mask.sum()
        if cls_count > 0:
            cls_correct = (predictions[cls_mask] == labels[cls_mask]).sum()
            cls_accuracy = cls_correct / cls_count
            inference_stats['class_accuracies'][str(cls)] = float(cls_accuracy)
        else:
            inference_stats['class_accuracies'][str(cls)] = 0.0
    
    # Calculate confusion matrix
    if len(labels) > 0:
        cm = confusion_matrix(labels, predictions)
        inference_stats['confusion_matrix'] = cm.tolist()
    else:
        inference_stats['confusion_matrix'] = []
    
    # Save statistics to JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stats_file = f"inference_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(inference_stats, f, indent=4)
    print(f"\n推理统计数据已保存到: {stats_file}")
    
    # Save confusion matrix to a separate JSON file
    cm_file = f"confusion_matrix_{timestamp}.json"
    cm_data = {'confusion_matrix': inference_stats['confusion_matrix']}
    try:
        class_names = dataloader.dataset.class_names
        cm_data['class_names'] = class_names
    except AttributeError:
        cm_data['class_names'] = [str(i) for i in range(len(inference_stats['class_accuracies']))]
    with open(cm_file, 'w') as f:
        json.dump(cm_data, f, indent=4)
    print(f"混淆矩阵已保存到: {cm_file}")
    
    return predictions, labels, accuracies, inference_stats

def main():
    """主函数，执行模型推理和结果可视化"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # Dataset parameters
    data_root = 'D:/mdx/HAIR_R.1/dataset'
    joint_mask_rate = 0.3
    frame_mask_rate = 0.2
    brightness_factor = 1.0
    
    # Load test dataset
    test_dataset = MyRGBSkeletonDataset(
        data_root_path=data_root,
        joint_mask_rate=joint_mask_rate,
        frame_mask_rate=frame_mask_rate,
        brightness_factor=brightness_factor,
        train=False  # Use test split
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    # Load model
    model_path = 'D:/mdx/HAIR_R.1/HAIR/pth/model_epoch_142_acc_0.9583_44.pth'
    model = load_model(model_path, device)
    
    # Perform inference
    predictions, labels, accuracies, inference_stats = inference(model, test_loader, device)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(accuracies) if accuracies else 0.0
    print(f'\n测试集总体准确率: {overall_accuracy:.4f}')
    
    # Print inference efficiency stats
    print(f'\n推理效率统计:')
    print(f'模型FLOPs: {inference_stats["flops"]}')
    print(f'总推理时间: {inference_stats["total_time"]:.4f} 秒')
    print(f'平均单个视频推理时间: {np.mean(inference_stats["per_video_time"]):.4f} 秒' if inference_stats["per_video_time"] else '无视频推理时间')
    
    # Find min/max inference time videos
    video_times = inference_stats['video_times']
    if video_times:
        min_time_video = min(video_times.items(), key=lambda x: x[1])
        max_time_video = max(video_times.items(), key=lambda x: x[1])
        print(f'最短推理时间视频: {min_time_video[0]}, 时间: {min_time_video[1]:.4f} 秒')
        print(f'最长推理时间视频: {max_time_video[0]}, 时间: {max_time_video[1]:.4f} 秒')
        
        # Visualize min/max inference time video frames
        try:
            plt.figure(figsize=(12, 5))
            min_time_idx = int(min_time_video[0].split('_')[1]) - 1
            min_time_frame = test_dataset.RGB_data[min_time_idx][0].numpy()
            if min_time_frame.shape[-1] == 3:  # Ensure RGB format
                min_time_frame = min_time_frame.transpose(1, 2, 0)  # Convert to HWC
            
            max_time_idx = int(max_time_video[0].split('_')[1]) - 1
            max_time_frame = test_dataset.RGB_data[max_time_idx][0].numpy()
            if max_time_frame.shape[-1] == 3:
                max_time_frame = max_time_frame.transpose(1, 2, 0)
            
            plt.subplot(121)
            plt.imshow(min_time_frame)
            plt.title(f'最短推理时间视频\n({min_time_video[1]:.4f}秒)')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(max_time_frame)
            plt.title(f'最长推理时间视频\n({max_time_video[1]:.4f}秒)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"可视化视频帧失败: {e}")
    
    # Visualize confusion matrix heatmap
    if inference_stats['confusion_matrix']:
        plt.figure(figsize=(10, 8))
        # Try to get class names from dataset, else use indices
        try:
            class_names = test_dataset.class_names
        except AttributeError:
            class_names = [str(i) for i in range(len(inference_stats['class_accuracies']))]
        
        confusion_matrix = np.array(inference_stats['confusion_matrix'])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('heat mapc')
        plt.xlabel('Prediced label')
        plt.ylabel('real label')
        plt.show()
    else:
        print("无混淆矩阵数据，无法生成热图")
    
    # Print per-class accuracies
    print("\n每个动作类的准确率:")
    try:
        class_names = test_dataset.class_names
    except AttributeError:
        class_names = None
    
    for cls, acc in inference_stats['class_accuracies'].items():
        cls_name = class_names[int(cls)] if class_names else cls
        print(f"动作 {cls_name}: {acc:.4f}")

if __name__ == '__main__':
    main()