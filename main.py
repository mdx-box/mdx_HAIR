import torch 
from torch import nn
import numpy as np
import torch.multiprocessing.spawn
import torch.utils
import torch.utils.data
from train_loop import train
from Multiview_Multimodal_model import Multi_view_modal_model
from torchinfo import summary
from utils.tool import load_config
from einops import rearrange, reduce
from Datasets_DataLoader.RGB_Skelelton_datasets import MyRGBSkeletonDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel 
import os
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device =  'cpu'
    paras_for_multi_view = load_config('MultiViewModel')
    paras_for_skeleton = load_config('SkeletonModel')
    paras_for_global_encoder = load_config('MultiViewModel')['paras_for_global_encoder']
    paras_for_overall_encoder = load_config('Overclasificationencoder')
    # 启动分布式训练环境
    train_dataset = MyRGBSkeletonDataset(data_root_path='D:/mdx/HAIR_R.1/dataset')
    print(f'train_dataset = {len(train_dataset)}')
    # train_dataloader = DataLoader(dataset=train_dataset,
    #                               batch_size=16,  # 直接设置batch_size为16
    #                               num_workers=4,
    #                               pin_memory=True,
    #                               shuffle=True,
    #                               collate_fn=None,
    #                               drop_last=True)
    model = Multi_view_modal_model(paras_for_multi_veiw=paras_for_multi_view,
                                   paras_for_skeleton=paras_for_skeleton,
                                      paras_for_global_encoder=paras_for_global_encoder,
                                   paras_for_overall_encoder=paras_for_overall_encoder,
                                   depths=16).to(device)
    
    # 使用torchinfo统计模型参数
    
    train_acc, test_acc = train(model=model,
                                  dataset=train_dataset,
                                  Epoches=150,
                                  batch_size=16,
                                  device=device)

if __name__ == '__main__':
    main()