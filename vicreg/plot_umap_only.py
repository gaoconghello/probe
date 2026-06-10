import numpy as np
import umap
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def plot_umap(features_h, true_labels=None, mode="kmeans", save_path="umap_result.png"):
    # 进行 L2 归一化，对齐自监督的夹角语义度量
    features_h = features_h / (np.linalg.norm(features_h, axis=1, keepdims=True) + 1e-8)
    
    print("开始计算 UMAP 降维...")
    print("采用激进参数: n_neighbors=50 (保留全局结构), min_dist=0.01 (极致挤压局部簇簇)...")
    
    # UMAP 核心参数设置
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=50, 
        min_dist=0.01, 
        metric='cosine', 
        random_state=42
    )
    features_2d = reducer.fit_transform(features_h)
    
    plt.figure(figsize=(12, 10))
    
    if mode == "kmeans":
        print("正在使用 K-Means 进行无监督着色...")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_h)
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels) if true_labels is not None else 0
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=15)
        plt.colorbar(scatter, ticks=range(10))
        plt.title(f"UMAP Visualization (K-Means Clusters)\nNMI = {nmi_score*100:.2f}%")
        
    elif mode == "supervised":
        print("正在使用真实标签进行监督着色...")
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.7, s=15)
        legend1 = plt.legend(*scatter.legend_elements(), title="Categories", loc="best")
        for i, text in enumerate(classes):
            legend1.get_texts()[i].set_text(f"{i}: {text}")
        plt.gca().add_artist(legend1)
        plt.colorbar(scatter, ticks=range(10))
        plt.title("UMAP Visualization (Colored by True Labels)")
        
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ UMAP 图像绘制完成！已保存为 '{save_path}'")

def main():
    data_file = "tsne_data.npz"
    if not os.path.exists(data_file):
        print(f"错误: 找不到特征数据 '{data_file}'！")
        return
        
    data = np.load(data_file)
    features_h = data["features_h"]
    labels = data["labels"]
    
    # 推荐使用 supervised 或者 kmeans 模式进行观测
    plot_umap(features_h, true_labels=labels, mode="supervised", save_path="umap_supervised.png")
    plot_umap(features_h, true_labels=labels, mode="kmeans", save_path="umap_kmeans.png")

if __name__ == "__main__":
    main()
