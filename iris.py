import numpy as np
import pandas as pd

np.random.seed(2026)

df = pd.read_csv("iris.csv", header=None)
data = df.iloc[:,:4].values
y_true_str = df.iloc[:,4].values # 真实标签

# 将字符串标签转换为数值标签 (0, 1, 2)，以便后续计算
# Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2 (顺序可能不同)
label_mapping = {label: idx for idx, label in enumerate(np.unique(y_true_str))}
y_true = np.array([label_mapping[label] for label in y_true_str])

# --- 2. 定义 K-Means 算法 ---
def kmeans_numpy(X, k, max_iters=100):
    # 1. 初始化中心点 (随机从数据中选取 k 个点)
    indices = np.random.choice(X.shape[0], k, replace=False)
    centers = X[indices]

    for _ in range(max_iters):
        # 2. 计算距离 (计算每个点到每个中心的欧氏距离)
        # 利用广播机制: (N, 1, D) - (1, K, D) -> (N, K, D) -> sum -> (N, K)
        distances = np.sqrt(((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))

        # 3. 分配簇 (找到距离最近的中心点的索引)
        labels = np.argmin(distances, axis=1)

        # 4. 更新中心点 (计算每个簇内所有点的均值)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 5. 检查收敛 (如果中心点不再变化，则停止)
        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return labels, centers


# --- 3. 运行算法 ---
k = 3
final_labels, final_centers = kmeans_numpy(data, k)

# --- 4. 结果展示 ---
print("最终聚类中心:\n", final_centers)


def get_purity_accuracy(y_true, y_pred, k):
    # 统计每个簇中各类别的数量
    cluster_counts = np.zeros((k, k), dtype=int)  # 假设真实类别也是k个
    true_classes = np.unique(y_true)

    # 构建混淆矩阵逻辑
    for i in range(k):
        # 获取属于簇 i 的样本的真实标签
        labels_in_cluster = y_true[y_pred == i]
        for t in true_classes:
            count = np.sum(labels_in_cluster == t)
            cluster_counts[i, t] = count

    # 计算正确分类的数量
    # 对每个簇，取最大值（即该簇对应的最可能的类别）
    correct_count = np.sum(np.max(cluster_counts, axis=1))
    accuracy = correct_count / len(y_true)
    return accuracy


accuracy = get_purity_accuracy(y_true, final_labels, k)
print(f"聚类准确率: {accuracy:.4f}")
