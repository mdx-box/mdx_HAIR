import torch 
from torch import nn
import numpy as np
import torch.multiprocessing.spawn
import torch.utils
import torch.utils.data
from train_loopRGB import RGB_train
# from Multiview_Multimodal_model import Multi_view_modal_model
#是将来自多视角和多模态的数据相互融合，并最终输出分类结果
from utils.tool import load_config
from einops import rearrange, reduce
from Datasets_DataLoader.RGB_Skelelton_datasets import MyRGBSkeletonDataset
from torch.utils.data import  DataLoader
from engine import MultiViewModel
from RGBDataloader import MyRGBDataset
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    paras_for_multi_view = load_config('MultiViewModel')
    paras_for_global_encoder = load_config('MultiViewModel')['paras_for_global_encoder']
    print(f"开始训练！")
    # 启动分布式训练环境
    train_dataset = MyRGBDataset(data_root_path='E:/mdx/SelfAttention_ActionRecognition/Input_dataset0')
    print(f'train_dataset = {len(train_dataset)}')

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=load_config('Batch_size'),
                                  num_workers=8,
                                  pin_memory=True,
                                  shuffle=True,
                                  collate_fn=None,
                                  drop_last=True)
    model = MultiViewModel(fusion_layer=paras_for_multi_view['fusion_layer'],
                           mlp_dims=paras_for_multi_view['mlp_dims'],
                           mlp_dropout=paras_for_multi_view['mlp_dropout'],
                           num_heads=paras_for_multi_view['num_heads'],
                           num_layer=paras_for_multi_view['num_layers'],
                           atten_dropout=paras_for_multi_view['atten_dropout'],
                           embedding_dims=paras_for_multi_view['embedding_dims'],
                           paras_for_global_encoder=paras_for_global_encoder,
                           view_number=[8,4,2])
    # print(f'model = {model}')
    train_loss, train_acc = RGB_train(model=model,
                                  train_dataloader=train_dataloader,
                                  Epoches=2000,
                                  device=device)
if __name__ == '__main__':
    main()
    