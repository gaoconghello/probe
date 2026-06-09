import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. 数据增强与数据集加载 (对齐客户最佳实践)
# -------------------------------------------------------------
class ContrastiveTransformations:
    def __init__(self):
        # 针对 FashionMNIST 的仿射变换数据增强设计，借鉴客户 @bk 项目中的成功经验
        self.base_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomAffine(degrees=25, translate=(0.2, 0.1), scale=(0.7, 1.3), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
        ])

    def __call__(self, x):
        # 返回同一张图片经过两次随机数据增强后的两个 View
        return [self.base_transforms(x), self.base_transforms(x)]

# -------------------------------------------------------------
# 2. VICReg 模型结构设计 (包含 2048 维大投影头)
# -------------------------------------------------------------
class VICRegModel(nn.Module):
    def __init__(self, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        # 获取基础 ResNet-18 并进行小图适配
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        
        # 提取 Encoder 部分 (去除最后的 fc 层) -> 输出 512 维特征 h
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 宽且深的 3 层 Projector (自监督学习特征保护的关键)
        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], -1) # 特征表达 h: 512 维
        z = self.projector(h)      # 投影特征 z: 2048 维
        return h, z

# -------------------------------------------------------------
# 3. VICReg 损失函数 (不变性、方差、协方差三大约束)
# -------------------------------------------------------------
def invariance_loss(z1, z2):
    # 不变性约束：MSE 损失，拉近同一张图两个增强版本的距离
    return F.mse_loss(z1, z2)

def variance_loss(z1, z2, gamma=1.0, eps=1e-4):
    # 方差约束：铰链损失，强行保证特征在 Batch 维度有足够的方差，防止模型塌陷
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))
    return std_loss

def covariance_loss(z1, z2):
    # 协方差约束：惩罚不同特征维度之间的协方差，强行让各个维度学到不同的独立特征
    N, D = z1.size()
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    
    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow(2).sum() / D + cov_z2[~diag.bool()].pow(2).sum() / D
    return cov_loss

class VICRegLoss(nn.Module):
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z1, z2):
        sim = invariance_loss(z1, z2)
        var = variance_loss(z1, z2)
        cov = covariance_loss(z1, z2)
        loss = self.sim_weight * sim + self.var_weight * var + self.cov_weight * cov
        return loss, sim, var, cov

# -------------------------------------------------------------
# 4. 可视化 t-SNE 函数 (带特征 L2 归一化)
# -------------------------------------------------------------
def plot_tsne(features_h, labels, save_path="tsne_result.png"):
    # 进行 L2 归一化，对齐自监督的夹角语义度量
    features_h = features_h / (np.linalg.norm(features_h, axis=1, keepdims=True) + 1e-8)
    
    print("开始进行 PCA 降维 (512 维 -> 50 维)...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_h)
    
    print("开始计算 t-SNE (50 维 -> 2 维)...")
    tsne = TSNE(n_components=2, perplexity=50, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    print(f"正在绘制并保存 t-SNE 图像至 {save_path}...")
    plt.figure(figsize=(12, 10))
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    
    legend1 = plt.legend(*scatter.legend_elements(), title="Categories", loc="best")
    for i, text in enumerate(classes):
        legend1.get_texts()[i].set_text(f"{i}: {text}")
    plt.gca().add_artist(legend1)
    
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE Visualization of VICReg Features (h)\nVICReg + Customized Augmentation")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("t-SNE 图像已保存完毕。")

# -------------------------------------------------------------
# 4.5 在线 KNN 评估函数
# -------------------------------------------------------------
@torch.no_grad()
def evaluate_knn(model, train_loader, test_loader, device, k=20, max_samples=2000):
    model.eval()
    
    # 提取训练集特征
    train_features, train_labels = [], []
    for x, y in train_loader:
        x = x.to(device)
        h, _ = model(x)
        h = F.normalize(h, dim=1) # L2 归一化
        train_features.append(h.cpu())
        train_labels.append(y)
        if len(torch.cat(train_features, dim=0)) >= max_samples:
            break
    train_features = torch.cat(train_features, dim=0)[:max_samples]
    train_labels = torch.cat(train_labels, dim=0)[:max_samples]
    
    # 提取测试集特征
    test_features, test_labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        h, _ = model(x)
        h = F.normalize(h, dim=1)
        test_features.append(h.cpu())
        test_labels.append(y)
        if len(torch.cat(test_features, dim=0)) >= max_samples:
            break
    test_features = torch.cat(test_features, dim=0)[:max_samples]
    test_labels = torch.cat(test_labels, dim=0)[:max_samples]
    
    # 送入 GPU 进行矩阵运算
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # 计算余弦相似度并投票
    sim_matrix = torch.matmul(test_features, train_features.T)
    _, topk_indices = sim_matrix.topk(k, dim=1)
    topk_labels = train_labels[topk_indices]
    predictions = torch.mode(topk_labels, dim=1).values
    
    correct = (predictions == test_labels).sum().item()
    accuracy = correct / len(test_labels)
    
    model.train()
    return accuracy

# -------------------------------------------------------------
# 5. 主训练与评估流程
# -------------------------------------------------------------
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU for acceleration", flush=True)
        print(f"CUDA版本: {torch.version.cuda}", flush=True)
        print(f"GPU设备: {torch.cuda.get_device_name(0)}", flush=True)
        # 开启 cuDNN 自动基准调优，自动寻找最高效的卷积算法
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU", flush=True)
    
    # 指向父目录下的 data 文件夹，避免重复下载
    data_dir = '../data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print("正在加载训练数据集...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ContrastiveTransformations()
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1024, # 降低到 1024，保证更多的梯度更新步数
        shuffle=True, 
        num_workers=16, # 发挥 128 线程撕裂者实力，暴力提速数据加载
        pin_memory=True, # 开启锁页内存加速
        persistent_workers=True, # 保持 worker 存活
        drop_last=True
    )
    
    # 归一化测试集 transform
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.2860, 0.2860, 0.2860), std=(0.3530, 0.3530, 0.3530))
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
        batch_size=1024, 
        shuffle=False, 
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    
    eval_train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform
    )
    eval_train_loader = DataLoader(
        eval_train_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化模型、优化器、余弦衰减学习率、损失函数
    model = VICRegModel(proj_hidden_dim=2048, proj_output_dim=2048).to(device)
    
    # AdamW 优化器，配置 1e-4 的权重衰减
    # 由于使用了较大的 1024 Batch Size，我们将学习率稍微降低到 5e-4 以保证训练稳定不爆炸
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    epochs = 1000 # 依据自监督学习的特性和客户最佳实践，将轮数提升至 1000
    # 余弦退火学习率调度器：在 1000 轮内将学习率平滑降至 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    # 对齐客户默认的权重分配：sim=25.0, var=25.0, cov=1.0
    criterion = VICRegLoss(sim_weight=25.0, var_weight=25.0, cov_weight=1.0)
    
    # 开启 AMP (自动混合精度) 加速
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"开始 VICReg 对比预训练，共 {epochs} 轮...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_sim = 0
        total_var = 0
        total_cov = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            # non_blocking=True 配合 pin_memory 实现异步数据传输
            x_i = images[0].to(device, non_blocking=True)
            x_j = images[1].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 使用 autocast 开启前向传播的混合精度加速
            with torch.cuda.amp.autocast():
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss, sim, var, cov = criterion(z_i, z_j)
                
            # 缩放 loss，反向传播并更新参数
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_sim += sim.item()
            total_var += var.item()
            total_cov += cov.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f} (Sim: {sim.item():.4f}, Var: {var.item():.4f}, Cov: {cov.item():.4f})")
                
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 动态 KNN 评估频率：前 200 轮每 50 轮评估，之后每 10 轮评估，以及最后一轮
        current_epoch = epoch + 1
        eval_flag = False
        if current_epoch <= 200 and current_epoch % 50 == 0:
            eval_flag = True
        elif current_epoch > 200 and current_epoch % 10 == 0:
            eval_flag = True
        elif current_epoch == epochs:
            eval_flag = True
            
        if eval_flag:
            print(f"Epoch [{current_epoch}/{epochs}] 结束，当前学习率: {current_lr:.6f}，正在进行在线 KNN 评估...", flush=True)
            knn_acc = evaluate_knn(model, eval_train_loader, test_loader, device, k=20)
            print(f"--- Epoch [{current_epoch}/{epochs}] 结束，平均 Loss: {avg_loss:.4f}，KNN 预测准确率: {knn_acc * 100:.2f}% ---", flush=True)
        else:
            print(f"--- Epoch [{current_epoch}/{epochs}] 结束，平均 Loss: {avg_loss:.4f} (跳过本轮 KNN 评估以加速训练) ---", flush=True)
        
    print("训练结束！正在提取特征进行 t-SNE 可视化...")
    
    # 提取测试集的 Encoder 特征 h 用于 t-SNE
    model.eval()
    features_h = []
    labels_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            h, _ = model(x)
            features_h.append(h.cpu())
            labels_list.append(y)
            if len(torch.cat(features_h, dim=0)) >= 2000:
                break
                
    features_h = torch.cat(features_h, dim=0)[:2000].numpy()
    labels = torch.cat(labels_list, dim=0)[:2000].numpy()
    
    # 保存训练好的模型权重
    torch.save(model.state_dict(), "vicreg_fashionmnist.pth")
    print("模型权重已保存为 vicreg_fashionmnist.pth")
    
    # 保存特征数据，方便后期直接调整 t-SNE 参数
    np.savez("tsne_data.npz", features_h=features_h, labels=labels)
    print("提取的特征数据已保存为 tsne_data.npz")
    
    # 绘制并保存 t-SNE 可视化图像
    plot_tsne(features_h, labels, save_path="tsne_result.png")

if __name__ == "__main__":
    main()
