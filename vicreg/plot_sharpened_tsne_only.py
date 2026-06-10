import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def sharpen_features(features, k=15, alpha=0.8, iterations=3):
    """
    高维特征锐化预处理 (High-dimensional Sharpening)
    将每个点向其 K 个最近邻的质心移动，增加簇间空白（簇内更紧密，簇间更疏远）
    """
    print(f"执行高维特征锐化预处理 (K={k}, Alpha={alpha}, Iterations={iterations})...")
    features_sharp = np.copy(features)
    for i_iter in range(iterations):
        print(f"  正在执行第 {i_iter+1}/{iterations} 轮锐化压缩...")
        nn = NearestNeighbors(n_neighbors=k, metric='cosine')
        nn.fit(features_sharp)
        distances, indices = nn.kneighbors(features_sharp)
        
        new_features = np.zeros_like(features_sharp)
        for i in range(len(features_sharp)):
            # 计算邻居的质心
            neighbor_center = np.mean(features_sharp[indices[i]], axis=0)
            # 向质心收缩
            new_features[i] = features_sharp[i] * (1 - alpha) + neighbor_center * alpha
            
        features_sharp = new_features
        # 保持在单位球面上
        features_sharp = features_sharp / (np.linalg.norm(features_sharp, axis=1, keepdims=True) + 1e-8)
        
    return features_sharp

def plot_sharpened_tsne(features_h, true_labels=None, mode="kmeans", save_path="sharpened_tsne_result.png"):
    # 第一次 L2 归一化
    features_h = features_h / (np.linalg.norm(features_h, axis=1, keepdims=True) + 1e-8)
    
    # 【核心逻辑】：执行高维锐化预处理，在降维前就强行拉开间距
    features_sharp = sharpen_features(features_h, k=20, alpha=0.8, iterations=3)
    
    print("开始进行 PCA 降维 (512 维 -> 50 维)...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_sharp)
    
    print("开始计算 Sharpened t-SNE (50 维 -> 2 维)...")
    # 继续叠加我们之前的早退夸张技巧
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=24.0, metric="cosine", init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    
    plt.figure(figsize=(12, 10))
    
    if mode == "kmeans":
        print("正在使用 K-Means 进行无监督着色...")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_h) # 注意：聚类评估依然使用原始特征，因为它是客观的
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels) if true_labels is not None else 0
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=15)
        plt.colorbar(scatter, ticks=range(10))
        plt.title(f"Sharpened t-SNE Visualization (K-Means Clusters)\nNMI = {nmi_score*100:.2f}%")
        
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
        plt.title("Sharpened t-SNE Visualization (Colored by True Labels)")
        
    plt.xlabel("Sharpened t-SNE Dimension 1")
    plt.ylabel("Sharpened t-SNE Dimension 2")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sharpened t-SNE 图像绘制完成！已保存为 '{save_path}'")

def main():
    data_file = "tsne_data.npz"
    if not os.path.exists(data_file):
        print(f"错误: 找不到特征数据 '{data_file}'！")
        return
        
    data = np.load(data_file)
    features_h = data["features_h"]
    labels = data["labels"]
    
    plot_sharpened_tsne(features_h, true_labels=labels, mode="supervised", save_path="sharpened_tsne_supervised.png")
    plot_sharpened_tsne(features_h, true_labels=labels, mode="kmeans", save_path="sharpened_tsne_kmeans.png")

if __name__ == "__main__":
    main()
