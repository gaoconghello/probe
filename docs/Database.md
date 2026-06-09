获取 FashionMNIST 数据集非常方便。对于绝大多数开发者来说，**最推荐的方式是通过代码框架直接自动下载**，而不是手动去网页上下载压缩包。

由于前面为你提供的方案是基于 PyTorch 的，以下我将重点介绍在代码中自动下载的写法，以及原文件的获取地址。

### 方法一：使用 PyTorch 自动下载（最推荐）

PyTorch 的 `torchvision` 库已经内置了 FashionMNIST 数据集。你只需要在代码中设置 `download=True`，它就会自动帮你下载并解压到指定的文件夹中。

你可以直接运行下面这段 Python 代码：

```python
import torchvision
import torchvision.transforms as transforms
import os

# 创建一个存放数据的文件夹
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("开始下载/加载训练集...")
train_dataset = torchvision.datasets.FashionMNIST(
    root=data_dir,       # 数据集下载后存放的相对路径
    train=True,          # True 表示获取训练集 (60,000 张图)
    download=True,       # 如果该路径下没有数据，则自动联网下载
    transform=transforms.ToTensor() # 将图片转换为 PyTorch 张量
)

print("开始下载/加载测试集...")
test_dataset = torchvision.datasets.FashionMNIST(
    root=data_dir,       
    train=False,         # False 表示获取测试集 (10,000 张图)
    download=True,       
    transform=transforms.ToTensor()
)

print(f"下载完成！数据保存在 {os.path.abspath(data_dir)} 目录下。")
print(f"训练集数量: {len(train_dataset)}, 测试集数量: {len(test_dataset)}")

```

运行完毕后，你的项目目录下会多出一个 `data/FashionMNIST/raw/` 文件夹，里面就是下载好的数据文件。后续再次运行这段代码时，框架检测到文件已存在，就不会重复下载了。

---

### 方法二：去官方 GitHub 仓库手动下载（原始文件）

如果你因为网络环境（如内网环境或下载极其缓慢）无法通过代码直接拉取，或者你想自己手动管理原始数据文件，可以去它的官方 GitHub 页面下载。

* **官方仓库地址：** [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

在仓库的 README 页面中，往下滑能看到 **"Get the Data"** 这一节，里面有四个 `.gz` 压缩包的直链：

1. `train-images-idx3-ubyte.gz` (训练集图片，约 26MB)
2. `train-labels-idx1-ubyte.gz` (训练集标签，约 29KB)
3. `t10k-images-idx3-ubyte.gz` (测试集图片，约 4.3MB)
4. `t10k-labels-idx1-ubyte.gz` (测试集标签，约 5KB)

**手动下载后的处理：**
如果你手动下载了这四个文件，请在你的项目根目录下依次创建文件夹 `data/FashionMNIST/raw/`，然后将这四个 `.gz` 文件原封不动地放进去（**不需要解压**）。之后再次运行 `torchvision.datasets.FashionMNIST(root='./data', download=True)` 时，PyTorch 就会直接读取本地文件。