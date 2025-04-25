import torch
from torch import nn
#是将来自多视角和多模态的数据相互融合，并最终输出分类结果
from RGB.engine import MultiViewModel, Transformer, MultiHeadSelfAttentionBlock
from Skeleton.Skele_transformer import ActionTransformer as SkeletonTransformer
from utils.tool import load_config
from einops import rearrange, reduce
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#定义全局输出编码器，需要注意的是：多视角模型的输入为：[view_number, batch_size, channels, depths, height, width],多视角模型的输出维度为：[batch_size,3*(nt+1)*nh*nw,embedding_of_globalEncoder ]
#骨骼模型的输入为[batch_size, depths, keypoints, channles], 骨骼模型的输出为：[batch_size, depths+1, embedding_dims]
class Fusion_transformer(nn.Module):

    def __init__(self, atten_dropout, num_heads, mlp_dim, mlp_dropout, embedding_dims):
        super(Fusion_transformer,self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.mutiheadattention = MultiHeadSelfAttentionBlock(embedding_dims=embedding_dims,
                                                       num_heads=num_heads,
                                                       attn_dropout=atten_dropout).to(device)
        self.layernorm = nn.LayerNorm(normalized_shape=embedding_dims).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dims,out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_dim,out_features=embedding_dims),
            nn.Dropout(p=mlp_dropout)
        ).to(device)
    def forward(self, query, key, value):
        #这里的输入是编码完成的toknes，其维度为[batch_size, nt*nh*nw, embedding_dims]
        # input = x
        x = self.mutiheadattention(query, key, value)
        x = self.layernorm(x + key)
        x = self.layernorm(x + self.mlp(x))
        return x

class Multi_view_modal_model(nn.Module):
    def __init__(self, paras_for_multi_veiw:dict, 
                 paras_for_skeleton:dict, 
                 paras_for_global_encoder:dict,
                 paras_for_overall_encoder:dict,
                 depths:int):
        super().__init__()
        self.depths = depths
        self.fusion_transformer = Fusion_transformer(atten_dropout=0.1,
                                                     mlp_dropout=0.1,
                                                     num_heads=8,
                                                     mlp_dim=256, embedding_dims=256)
        self.multiviewmodel = MultiViewModel(fusion_layer=paras_for_multi_veiw['fusion_layer'],
                                             mlp_dims=paras_for_multi_veiw['mlp_dims'],
                                             mlp_dropout=paras_for_multi_veiw['mlp_dropout'],
                                             num_heads=paras_for_multi_veiw['num_heads'],
                                             num_layer=paras_for_multi_veiw['num_layers'],
                                             atten_dropout=paras_for_multi_veiw['atten_dropout'],
                                             embedding_dims=paras_for_multi_veiw['embedding_dims'],
                                             paras_for_global_encoder=paras_for_global_encoder,
                                             view_number=[2,4,8])
        self.skeletontmodel = SkeletonTransformer(d_model=paras_for_skeleton['embedding_dims'],
                             d_ff=paras_for_skeleton['mlp_dims'],
                             seq_length=paras_for_skeleton['depths'],
                             n_head=paras_for_skeleton['num_heads'],
                             dropout=0.2,
                             keypoints=paras_for_skeleton['keypoints'],
                             channels=paras_for_skeleton['channels'],
                             encoder_layer=paras_for_skeleton['num_layers'])
        self.overalltransformer = Transformer(atten_dropout=paras_for_overall_encoder['atten_dropout'],
                                       num_heads=paras_for_overall_encoder['num_heads'],
                                       mlp_dim=paras_for_overall_encoder['mlp_dim'],
                                       mlp_dropout=paras_for_overall_encoder['mlp_dropout'],
                                       embedding_dims=paras_for_overall_encoder['embedding_dims'])
        self.ff1 = nn.Linear(in_features=256, out_features=256)
        self.ff2 = nn.Linear(in_features=256, out_features=9)
        #维度转换
        self.ff3 = nn.Linear(in_features=17, out_features=980)
    def forward(self, x_for_multiview, x_for_skeleton):
        #注意multiview的全局编码器和骨架模型的输出编码器的维度要保持一致，否则需要额外的维度转换
        #这里的输入是来自多视角和骨骼模型的输出的融合，至于具体怎么融合，需要考虑一下
        # print(f'MMM:{x_for_multiview[0].shape},{x_for_multiview[1].shape}')
        y_from_multiview = self.multiviewmodel(x_for_multiview) #torch.Size([16, 441, 12])
        y_from_skeleton = self.skeletontmodel(x_for_skeleton) #torch.Size([16, 9, 12])
        # print(y_from_multiview.shape, y_from_skeleton.shape)
        # print(f'y_from_multiview:{y_from_multiview.shape},  y_from_skeleton:{ y_from_skeleton.shape}')
        y_from_skeleton = rearrange(self.ff3(rearrange(y_from_skeleton, 'b n c -> b c n')), 'b c n -> b n c')
        # print(f'y_from_multiview:{y_from_multiview.shape},  y_from_skeleton:{ y_from_skeleton.shape}')
        fused_input = self.fusion_transformer(y_from_skeleton, y_from_multiview, y_from_multiview)
        
        # fusion_input = torch.cat((y_from_multiview,y_from_skeleton),dim=1) #torch.Size([16, 450, 12]
        #进行维度转换
        #输出的维度为[batch_size, depths, classifier_number],其中classifier_number为每一个视频钟对应帧数的labels
        out = self.ff1(fused_input)
        out = self.ff2(out)
        return out[:,0,:]

# 测试
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paras_for_multi_view = load_config('MultiViewModel')
    paras_for_skeleton = load_config('SkeletonModel')
    paras_for_global_encoder = load_config('MultiViewModel')['paras_for_global_encoder']
    paras_for_overall_encoder = load_config('Overclasificationencoder')
    batch_size = load_config("Batch_size")
    print("\n=== 模型配置参数 ===")
    print(f"多视角模型参数: {paras_for_multi_view}")
    print(f"骨骼模型参数: {paras_for_skeleton}")
    print(f"全局编码器参数: {paras_for_global_encoder}")
    print(f"整体编码器参数: {paras_for_overall_encoder}")
    
    #多视角输入
    view1 = torch.randn(size=(batch_size,3,16,112,112)).to(device)
    view2 = torch.randn(size=(batch_size,3,16,112,112)).to(device)
    view3 = torch.randn(size=(batch_size,3,16,112,112)).to(device)
    x1 = [view1, view2, view3]
    #骨骼模型输入
    x2 = torch.randn(size=(batch_size,16,33,3)).to(device)
    
    print("\n=== 输入数据维度 ===")
    print(f"多视角输入维度: {[v.shape for v in x1]}")
    print(f"骨骼输入维度: {x2.shape}")
    
    #构建模型
    model = Multi_view_modal_model(paras_for_multi_veiw=paras_for_multi_view,
                                   paras_for_skeleton=paras_for_skeleton,
                                   paras_for_global_encoder=paras_for_global_encoder,
                                   paras_for_overall_encoder=paras_for_overall_encoder,
                                   depths=16).to(device)
    
    print("\n=== 模型结构 ===")
    print("多视角模型:")
    print(model.multiviewmodel)
    print("\n骨骼模型:")
    print(model.skeletontmodel)
    print("\n融合Transformer:")
    print(model.fusion_transformer)
    
    print("\n=== 前向传播 ===")
    y_from_multiview = model.multiviewmodel(x1)
    print(f"多视角模型输出维度: {y_from_multiview.shape}")
    
    y_from_skeleton = model.skeletontmodel(x2)
    print(f"骨骼模型输出维度: {y_from_skeleton.shape}")
    
    y_from_skeleton = rearrange(model.ff3(rearrange(y_from_skeleton, 'b n c -> b c n')), 'b c n -> b n c')
    print(f"维度转换后的骨骼输出: {y_from_skeleton.shape}")
    
    fused_input = model.fusion_transformer(y_from_skeleton, y_from_multiview, y_from_multiview)
    print(f"融合后的特征维度: {fused_input.shape}")
    
    out = model.ff2(model.ff1(fused_input))
    print(f"\n最终输出维度: {out[:,0,:].shape}")
    print("模型结构分析完成！")
    