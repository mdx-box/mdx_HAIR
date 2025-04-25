#模型训练和测试
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
import torch
from einops import rearrange

def test(model, test_dataloader, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    batch_count = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for item, (X_rgb, X_Skeleton, y) in enumerate(test_dataloader):
            X_rgb, X_Skeleton, y = X_rgb.to(torch.float), X_Skeleton.to(torch.float), y.to(torch.float)
            X_rgb, X_Skeleton, y = rearrange(X_rgb,'b t h w c -> b c t h w').to(device), X_Skeleton.to(device), y.to(device)
            
            y_pred_logit = model([X_rgb,X_rgb,X_rgb], X_Skeleton)
            y_index = y.argmax(1)
            loss = loss_fn(y_pred_logit, y_index)
            acc = (y_pred_logit.softmax(dim=1).argmax(dim=1) == y.argmax(1)).sum() / (y.shape[0])
            
            test_loss += loss.item()
            test_acc += acc.item()
            batch_count += 1
            
    avg_test_loss = test_loss / batch_count
    avg_test_acc = test_acc / batch_count
    return avg_test_loss, avg_test_acc

def train(model:torch.nn.Module,
        dataset,
        batch_size:int,
        Epoches:int,
        device):
    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print("Start training!!")
    # train_dataloader.sampler.set_e
    optimizer = torch.optim.SGD(params=model.parameters(),lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(Epoches):
        print(f'第{i}轮训练开始...')
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        for item, (X_rgb, X_Skeleton, y) in tqdm(enumerate(train_dataloader)):
            torch.cuda.empty_cache()
            # print(X_rgb.shape, X_Skeleton.shape)
            X_rgb, X_Skeleton, y = X_rgb.to(torch.float), X_Skeleton.to(torch.float), y.to(torch.float)
            X_rgb, X_Skeleton, y = rearrange(X_rgb,'b t h w c -> b c t h w').to(device), X_Skeleton.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred_logit = model([X_rgb,X_rgb,X_rgb], X_Skeleton)
            # 将one-hot编码转换为类别索引
            y_index = y.argmax(1)
            loss = loss_fn(y_pred_logit, y_index)
            loss.backward()
            optimizer.step()
            acc = (y_pred_logit.softmax(dim=1).argmax(dim=1) == y.argmax(1)).sum() / (y.shape[0])
            
            # 累积每个batch的loss和accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batch_count += 1
            
            # 每个batch直接打印loss和accuracy
            print(f'batch:{item}, loss:{loss.item():.4f}, acc:{acc:.4f}')
            
            # # 保存训练记录
            # with open('D:/mdx/HAIR_R.1/model_large.txt', 'a+') as f:
            #     f.write(f'{loss.item():.4f} {acc:.4f}\n')
                
            # 清理显存
            del y_pred_logit
            torch.cuda.empty_cache()
        
        # 计算并输出每个epoch的平均loss和accuracy
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_acc = epoch_acc / batch_count
        print(f'Epoch {i} 平均loss: {avg_epoch_loss:.4f}, 平均accuracy: {avg_epoch_acc:.4f}')
        
        # 在每个epoch结束后进行测试
        test_loss, test_acc = test(model, test_dataloader, device)
        print(f'Epoch {i} 测试集 loss: {test_loss:.4f}, accuracy: {test_acc:.4f}')
        
        # 保存每个epoch的训练和测试结果
        with open('D:/mdx/HAIR_R.1/HAIR/Results/modelm_81_fusion.txt', 'a+') as f:
            f.write(f'Epoch {i}:\n训练集 - loss: {avg_epoch_loss:.4f}, accuracy: {avg_epoch_acc:.4f}\n测试集 - loss: {test_loss:.4f}, accuracy: {test_acc:.4f}\n\n')
            
        # 当测试集准确率大于95%且是偶数epoch时保存模型
        if test_acc > 0.95 and i % 2 == 0:
            import os
            save_dir = 'D:/mdx/HAIR_R.1/HAIR/pth'
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f'model_epoch_{i}_acc_{test_acc:.4f}_44.pth')
            torch.save(model, save_path)
            print(f'模型已保存到: {save_path}')

    return avg_epoch_acc, test_acc