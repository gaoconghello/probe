import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import os

def plot_tsne_3d(features_h, true_labels, save_path="tsne_3d_interactive.html"):
    # 1. 特征 L2 归一化
    features_h = features_h / (np.linalg.norm(features_h, axis=1, keepdims=True) + 1e-8)
    
    print("开始进行 PCA 降维 (512 维 -> 50 维)...")
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features_h)
    
    print("开始计算 3D t-SNE (50 维 -> 3 维)...")
    # 关键修改：n_components 改为 3
    tsne = TSNE(
        n_components=3, 
        perplexity=50, 
        early_exaggeration=24.0, 
        metric="cosine", 
        init='pca', 
        random_state=42
    )
    features_3d = tsne.fit_transform(features_pca)
    
    print("正在生成可交互的 3D HTML 图像...")
    
    # 类别映射
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_names = [classes[label] for label in true_labels]
    
    # 构建 DataFrame 供 Plotly 渲染
    df = pd.DataFrame({
        'X': features_3d[:, 0],
        'Y': features_3d[:, 1],
        'Z': features_3d[:, 2],
        'Class': class_names
    })
    
    # 使用 Plotly 绘制 3D 散点图
    fig = px.scatter_3d(
        df, x='X', y='Y', z='Z', 
        color='Class', 
        opacity=0.8,
        title="Interactive 3D t-SNE Visualization (VICReg Features)"
    )
    
    # 调整点的大小，让图表看起来更精致
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'))
    
    # 导出为独立可交互的 HTML 网页文件
    fig.write_html(save_path)
    print(f"✅ 3D 交互网页生成完毕！请用浏览器直接双击打开 '{save_path}' 进行自由旋转和缩放查看。")

def main():
    data_file = "tsne_data.npz"
    if not os.path.exists(data_file):
        print(f"错误: 找不到特征数据 '{data_file}'！")
        return
        
    data = np.load(data_file)
    features_h = data["features_h"]
    labels = data["labels"]
    
    plot_tsne_3d(features_h, true_labels=labels, save_path="tsne_3d_interactive.html")

if __name__ == "__main__":
    main()
