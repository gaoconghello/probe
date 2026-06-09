import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def plot_tsne(features_h, labels, perplexity=50, save_path="tsne_result.png"):
    print("开始进行 PCA 降维 (512 维 -> 50 维)...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_h)
    
    print(f"开始计算 t-SNE (50 维 -> 2 维, perplexity={perplexity})...")
    # 使用 PCA 初始化提高稳定性，perplexity 可用于调整聚类紧凑度
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    print(f"正在绘制并保存 t-SNE 图像至 {save_path}...")
    plt.figure(figsize=(12, 10))
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 绘制散点图
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
    
    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(), title="Categories", loc="best")
    for i, text in enumerate(classes):
        legend1.get_texts()[i].set_text(f"{i}: {text}")
    plt.gca().add_artist(legend1)
    
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f"t-SNE Visualization of Encoder Features (h)\nSimCLR (perplexity={perplexity})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE 图像绘制完成！已保存为 '{save_path}'")

def main():
    data_file = "tsne_data.npz"
    if not os.path.exists(data_file):
        print(f"错误: 找不到特征数据文件 '{data_file}'！请先运行训练脚本生成数据。")
        return
        
    print(f"正在读取 '{data_file}'...")
    data = np.load(data_file)
    features_h = data["features_h"]
    labels = data["labels"]
    print(f"读取成功。特征形状: {features_h.shape}, 标签形状: {labels.shape}")
    
    # ---------------------------------------------------------
    # 💡 提示：你可以在下面直接修改 perplexity (例如 30, 40, 50, 80)
    # 并重新运行该脚本，即可在毫秒级内生成新的 t-SNE 图像，无需重新训练！
    # ---------------------------------------------------------
    perplexity = 50
    plot_tsne(features_h, labels, perplexity=perplexity, save_path="tsne_result.png")

if __name__ == "__main__":
    main()
