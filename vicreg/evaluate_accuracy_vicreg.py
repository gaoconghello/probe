import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从 train_vicreg_and_tsne import VICRegModel
from train_vicreg_and_tsne import VICRegModel

# -------------------------------------------------------------
# 1. 线性探针评估 (Linear Probe)
# -------------------------------------------------------------
def evaluate_linear_probe(device, model, train_loader, test_loader, epochs=100):
    print("\n=== 开始进行线性评估 (Linear Probe) ===")
    print("正在冻结 Encoder 权重，仅训练一个线性分类器...")
    
    # 冻结 Encoder 的所有权重参数
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # 定义简单的线性分类层 (512 维特征输入 -> 10 分类输出)
    classifier = nn.Linear(512, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练线性分类器
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # 使用冻结 of Encoder 提取特征
            with torch.no_grad():
                h = model.encoder(x)
                h = h.view(h.shape[0], -1) # [Batch, 512]
            
            # 分类器预测
            outputs = classifier(h)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_acc = correct / total
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc*100:.2f}%")
        
    # 测试分类器在测试集上的最终准确率
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            h = model.encoder(x)
            h = h.view(h.shape[0], -1)
            outputs = classifier(h)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    final_acc = correct / total
    print(f"👉 线性分类器在测试集上的准确率 (Linear Probe Accuracy): {final_acc*100:.2f}%")
    return final_acc

# -------------------------------------------------------------
# 2. 全量 KNN 评估 (Offline Weighted KNN)
# -------------------------------------------------------------
@torch.no_grad()
def evaluate_full_knn(device, model, train_loader, test_loader, k=20, batch_size=512):
    print("\n=== 开始进行全量测试集 KNN 评估 ===")
    model.eval()
    
    # 提取训练集所有样本的特征向量和标签
    train_features = []
    train_labels = []
    print("正在提取训练集全量特征向量...")
    for x, y in tqdm(train_loader):
        x = x.to(device)
        h, _ = model(x)
        h = h / (torch.norm(h, dim=1, keepdim=True) + 1e-8)
        train_features.append(h.cpu())
        train_labels.append(y)
        
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    # 提取测试集所有样本的特征向量和标签
    test_features = []
    test_labels = []
    print("正在提取测试集全量特征向量...")
    for x, y in tqdm(test_loader):
        x = x.to(device)
        h, _ = model(x)
        h = h / (torch.norm(h, dim=1, keepdim=True) + 1e-8)
        test_features.append(h.cpu())
        test_labels.append(y)
        
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # 利用矩阵相乘计算余弦相似度并预测
    correct = 0
    total = 0
    print("正在计算 KNN 距离与相似度投票...")
    
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    
    for i in range(0, len(test_features), batch_size):
        batch_feat = test_features[i:i+batch_size].to(device)
        batch_lab = test_labels[i:i+batch_size].to(device)
        
        sim_matrix = torch.matmul(batch_feat, train_features.T)
        _, topk_indices = sim_matrix.topk(k, dim=1)
        topk_labels = train_labels[topk_indices]
        predictions = torch.mode(topk_labels, dim=1).values
        
        correct += (predictions == batch_lab).sum().item()
        total += batch_lab.size(0)
        
    knn_acc = correct / total
    print(f"👉 全量测试集 KNN (k={k}) 评估 Accuracy: {knn_acc*100:.2f}%")
    return knn_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")
    
    # 加载模型
    model = VICRegModel(proj_hidden_dim=2048, proj_output_dim=2048)
    ckpt_path = "vicreg_fashionmnist.pth"
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到权重文件 '{ckpt_path}'！请先运行训练脚本 train_vicreg_and_tsne.py")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    print(f"成功加载已训练模型权重: {ckpt_path}")
    
    # 指向父目录下的 data 文件夹，避免重复下载
    data_dir = '../data'
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # 1. 评估全量 KNN 准确率
    evaluate_full_knn(device, model, train_loader, test_loader, k=20)
    
    # 2. 评估线性分类器（Linear Probe）准确率，提升到 100 轮对齐客户
    train_loader_shuffle = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    evaluate_linear_probe(device, model, train_loader_shuffle, test_loader, epochs=100)

if __name__ == "__main__":
    main()
