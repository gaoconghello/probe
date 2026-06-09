import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import os

def plot_tsne(features_h, true_labels=None, mode="kmeans", perplexity=50, save_path="tsne_result.png"):
    # 进行 L2 归一化，对齐自监督的夹角语义度量
    features_h = features_h / (np.linalg.norm(features_h, axis=1, keepdims=True) + 1e-8)
    
    print("开始进行 PCA 降维 (512 维 -> 50 维)...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_h)
    
    print(f"开始计算 t-SNE (50 维 -> 2 维, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    plt.figure(figsize=(12, 10))
    
    if mode == "kmeans":
        print("【无监督模式】正在使用 K-Means 对特征点进行无监督聚类 (K=10)...")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_h)
        
        if true_labels is not None:
            nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
            print(f"👉 无监督聚类评估：NMI (Normalized Mutual Information) = {nmi_score*100:.2f}%")
        
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=15)
        plt.colorbar(scatter, ticks=range(10))
        plt.title(f"Unsupervised t-SNE Visualization (Colored by K-Means Clusters)\nVICReg + K-Means (NMI = {nmi_score*100:.2f}%)")
        print(f"图像已着色（按 K-Means 自动生成的聚类 ID：0-9）。")

    elif mode == "uncolored":
        print("【单色分布模式】不使用任何标签/聚类着色，展示高维空间的天然物理聚簇形状...")
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c='#4682B4', alpha=0.6, s=15)
        plt.title(f"Pure Unsupervised t-SNE Feature Space Geometry\nVICReg Representation (No Labels / No Colors)")
        print(f"图像已着色（全单色展示）。")

    elif mode == "supervised":
        print("【监督着色模式】使用数据集的真实分类进行染色（仅供内部验证）...")
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.7, s=15)
        legend1 = plt.legend(*scatter.legend_elements(), title="Categories", loc="best")
        for i, text in enumerate(classes):
            legend1.get_texts()[i].set_text(f"{i}: {text}")
        plt.gca().add_artist(legend1)
        plt.colorbar(scatter, ticks=range(10))
        plt.title(f"Supervised t-SNE Visualization (Colored by True Labels)\nFor Validation Only (perplexity={perplexity})")

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
    # 💡 客户演示模式设置 (您可以修改下面的 mode 和 perplexity)
    # - mode="kmeans"    : 完全无监督聚类染色（最符合客户不体现真实标签的要求）
    # - mode="uncolored" : 完全单色物理分簇展示（纯粹展示几何分簇）
    # - mode="supervised" : 真实标签染色演示
    # ---------------------------------------------------------
    mode = "kmeans"  # 可选："kmeans", "uncolored", "supervised"
    perplexity = 50
    
    plot_tsne(
        features_h=features_h, 
        true_labels=labels, 
        mode=mode, 
        perplexity=perplexity, 
        save_path=f"tsne_result_{mode}.png"
    )

if __name__ == "__main__":
    main()
