import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. 数据增强与数据集加载 (基于 docs/Database.md 方法一)
# -------------------------------------------------------------
class ContrastiveTransformations:
    def __init__(self):
        # 针对 28x28 灰度图的温和数据增强设计，防止过度裁剪丢失关键特征
        self.base_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)), # 裁剪比例限制在 0.8-1.0
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # 单通道归一化
        ])

    def __call__(self, x):
        # 返回同一张图片经过两次随机数据增强后的正样本对
        return [self.base_transforms(x), self.base_transforms(x)]

# -------------------------------------------------------------
# 2. SimCLR 模型结构设计 (基于 docs/PRD.md & docs/t-SNE.md)
# -------------------------------------------------------------
class SimCLRModel(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # 获取基础 ResNet-18 并进行小图适配魔改
        # 1. 灰度图输入通道改为 1，卷积核改为 3x3，步长改为 1，padding 改为 1
        # 2. 移除 maxpool，避免 28x28 图片被过度降采样
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        
        # 提取 Encoder 部分 (去除最后的 fc 层)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 非线性 Projection Head (MLP)，仅用于计算对比损失，帮助特征 h 更好地保留原始语义
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], -1) # 特征表达 h: 512 维
        z = self.projector(h)      # 对比投影 z: 128 维
        return h, z

# -------------------------------------------------------------
# 3. 对比损失函数 InfoNCE Loss (基于 docs/t-SNE.md)
# -------------------------------------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1): # 较低的温度系数让困难负样本惩罚更大，使特征聚集更紧密
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        device = z_i.device
        batch_size = z_i.shape[0]
        
        # L2 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 合并所有特征 [2*N, D]
        representations = torch.cat([z_i, z_j], dim=0)
        
        # 计算余弦相似度矩阵 [2*N, 2*N]
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # 温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        # 掩盖自相似性 (对角线设为极小值)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(self_mask, -9e15)
        
        # 构建 Targets: i 与 i+N 互为正样本，i+N 与 i 互为正样本
        targets = torch.arange(2 * batch_size, device=device)
        targets[:batch_size] += batch_size
        targets[batch_size:] -= batch_size
        
        # 使用交叉熵计算损失
        loss = F.cross_entropy(similarity_matrix, targets)
        return loss

# -------------------------------------------------------------
# 4. 可视化 t-SNE 函数 (基于 docs/t-SNE.md 降维策略)
# -------------------------------------------------------------
def plot_tsne(features_h, labels, save_path="tsne_result.png"):
    print("开始进行 PCA 降维 (512 维 -> 50 维)...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_h)
    
    print("开始计算 t-SNE (50 维 -> 2 维)...")
    # 调高困惑度 (perplexity=50)，使用稳定稳定的 PCA 进行初始化
    tsne = TSNE(n_components=2, perplexity=50, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    print(f"正在绘制并保存 t-SNE 图像至 {save_path}...")
    plt.figure(figsize=(12, 10))
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 按不同类别绘制散点图
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    
    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(), title="Categories", loc="best")
    for i, text in enumerate(classes):
        legend1.get_texts()[i].set_text(f"{i}: {text}")
    plt.gca().add_artist(legend1)
    
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE Visualization of Encoder Features (h)\nSimCLR + Customized Augmentation")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("t-SNE 图像已保存完毕。")

# -------------------------------------------------------------
# 5. 主训练与评估流程
# -------------------------------------------------------------
def main():
    # 检测 GPU 加速环境并输出详细信息
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU for acceleration", flush=True)
        print(f"CUDA版本: {torch.version.cuda}", flush=True)
        print(f"GPU设备: {torch.cuda.get_device_name(0)}", flush=True)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for acceleration", flush=True)
    else:
        device = torch.device("cpu")
        print("Neither GPU nor MPS available. Using CPU", flush=True)
    
    # 5.1 数据准备 (docs/Database.md 第一种方式)
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print("正在加载训练数据集...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ContrastiveTransformations() # 使用对比增强
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )
    
    # 用于 t-SNE 可视化的测试集
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("正在加载测试数据集...", flush=True)
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=0
    )

    # 5.2 初始化模型、优化器和损失函数
    model = SimCLRModel(out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = InfoNCELoss(temperature=0.1) # 较低的温度系数 (0.1) 促使类内聚拢
    
    epochs = 10  # 为了快速展示，默认训练 10 轮。可根据需要增加轮数以获得更好的聚类效果
    print(f"开始对比训练，共 {epochs} 轮...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            # images 是由 ContrastiveTransformations 返回的两个增强样本 [img1, img2]
            x_i = images[0].to(device)
            x_j = images[1].to(device)
            
            optimizer.zero_grad()
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch [{epoch+1}/{epochs}] 结束，平均 Loss: {avg_loss:.4f} ---")
        
    print("训练结束！正在提取特征进行 t-SNE 可视化...")
    
    # 5.3 提取测试集的 Encoder 特征 h
    model.eval()
    features_h = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            h, _ = model(x)
            features_h.append(h.cpu())
            labels_list.append(y)
            # 为了绘制速度及图表清晰度，提取前 2000 个测试样本进行降维
            if len(torch.cat(features_h, dim=0)) >= 2000:
                break
                
    features_h = torch.cat(features_h, dim=0)[:2000].numpy()
    labels = torch.cat(labels_list, dim=0)[:2000].numpy()
    
    # 保存训练好的模型权重
    torch.save(model.state_dict(), "simclr_fashionmnist.pth")
    print("模型权重已保存为 simclr_fashionmnist.pth")
    
    # 保存提取的特征向量和标签，方便后期直接调整 t-SNE 参数
    np.savez("tsne_data.npz", features_h=features_h, labels=labels)
    print("提取的特征数据已保存为 tsne_data.npz")
    
    # 5.4 绘制并保存 t-SNE 可视化图像
    plot_tsne(features_h, labels, save_path="tsne_result.png")

if __name__ == "__main__":
    main()
