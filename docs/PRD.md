Linear Probe 准确率高达 92.29%，但在 t-SNE 上却无法形成清晰的类内聚集，这是一个在自监督学习（尤其是对比学习）中非常经典的现象。

这通常说明你的模型已经学到了**线性可分**的特征（即可以用一个超平面把不同类别的衣服切开），但并没有在特征空间里形成高密度的**高斯簇**。要解决“上衣无法聚成一簇”的问题，核心在于**针对小尺寸单通道图像优化数据增强**，以及**调整特征提取与降维的策略**。

以下我为你整理的一套基于 SimCLR 思想的完整落地方案，专门针对 FashionMNIST 这种 28x28 的灰度数据集进行了调优。

---

### 第一步：重构数据增强（破局的关键）

对比学习极其依赖数据增强来构造正样本对。很多开发者会直接照搬 ImageNet 的增强策略（如强烈的颜色抖动、大比例的随机裁剪），这对于 28x28 的 FashionMNIST 往往是毁灭性的。比如把一件上衣裁剪得只剩一个领口，模型就彻底认不出它了。

**FashionMNIST 专属增强策略：**

* **弃用：** `ColorJitter`（灰度图不需要）、比例过小的 `RandomResizedCrop`。
* **保留/修改：** 温和的裁剪、水平翻转、轻微的随机旋转、高斯模糊。

```python
import torchvision.transforms as transforms

# 构建一对正样本的 Transform
class ContrastiveTransformations:
    def __init__(self):
        self.base_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)), # 裁剪比例不能太小
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # FashionMNIST 是单通道
        ])

    def __call__(self, x):
        return [self.base_transforms(x), self.base_transforms(x)]

```

### 第二步：调整 Encoder 与 Projection Head

如果直接用标准的 ResNet-18，它的第一层卷积是 `7x7` 步长为 2，并且带有一个 `MaxPool`。这会把 28x28 的图片直接降采样到 7x7，丢失大量细节。

1. **修改 Encoder 输入层：** 将第一层卷积改为 `3x3`，步长 1，并去掉最大池化层。
2. **明确 $h$ 与 $z$ 的区别：**
* **$h$ (Representation):** Encoder 的直接输出（通常是 512 维）。**Linear Probe 和 t-SNE 必须使用 $h$。**
* **$z$ (Projection):** 经过 MLP 投射头后的输出。这部分仅用于计算对比损失，计算完直接丢弃。



```python
import torch.nn as nn
from torchvision.models import resnet18

class SimCLRModel(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # 1. 获取基础 ResNet，并做小图魔改
        resnet = resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # 改为单通道 3x3
        resnet.maxpool = nn.Identity() # 移除 maxpool
        
        # 提取 Encoder
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. 构建 Projection Head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], -1) # 得到 512 维特征 h
        z = self.projector(h)      # 得到降维特征 z (用于 InfoNCE loss)
        return h, z

```

### 第三步：对比学习训练调优 (InfoNCE Loss)

在训练时，除了常规的 Adam 优化器和学习率衰减，你需要特别关注温度系数 $\tau$ (Temperature)。

* **如果聚类不紧凑：** 尝试稍微**调低**温度系数 $\tau$（例如从 0.5 降到 0.1）。更低的 $\tau$ 会迫使模型对困难负样本（比如外观相似的T恤和套衫）进行更严苛的惩罚，从而把同类推得更近，异类拉得更远。

### 第四步：Linear Probe 验证

这一步你的准确率已经很高，说明逻辑没问题。为了严谨，标准做法是：

1. 冻结整个 `self.encoder` 的参数。
2. 在后面接一个简单的 `nn.Linear(512, 10)` 进行多分类交叉熵训练。
3. 输入仅使用基础的标准预处理（`ToTensor` 和 `Normalize`），不要加前面的强数据增强。

### 第五步：降维与 t-SNE 可视化策略

很多时候 t-SNE 画出来是一锅粥，并不全是模型的锅，而是降维策略的问题。高维特征直接做 t-SNE 效果通常不好。

**标准的可视化流线：** `提取特征 h -> PCA (降至约 50 维) -> t-SNE (降至 2 维)`

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设 features_h 是从验证集中提取的所有 h 特征 (Shape: [N, 512])
# 假设 labels 是对应的真实标签 (Shape: [N])

def plot_tsne(features_h, labels):
    # 1. 先用 PCA 降维去除冗余噪声，保留主要方差
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_h)
    
    # 2. t-SNE 核心计算
    # perplexity 困惑度：对于 FashionMNIST 这种密集数据，可以适当调高 (默认30，可尝试 40-50)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    # 3. 绘图 (按不同类别着色)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE Visualization of Encoder Features (h)")
    plt.show()

```

---

为了帮我们进一步定位问题所在：你目前在提取特征做可视化时，取的是 Encoder 骨干网络的直接输出 $h$，还是经过了 Projection Head 映射后的特征 $z$？