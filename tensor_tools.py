import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score


def align_to_y_axis(data):
    """
    Align the first node with respect to the 14th node to the Y-axis.
    This involves calculating the angle between these two nodes and
    rotating all nodes by this angle so that the line connecting
    the first and the 14th node aligns with the Y-axis.

    Parameters:
    - data (numpy.ndarray): The input data with shape (41390, 3, 82, 19),
                            where 3 represents X, Y, and confidence interval.

    Returns:
    - numpy.ndarray: The aligned data with the same shape as input.
    """

    # 获取第1个和第14个点的X, Y坐标
    x_1 = data[:, 0, :, 0]
    y_1 = data[:, 1, :, 0]
    x_14 = data[:, 0, :, 13]
    y_14 = data[:, 1, :, 13]

    # 计算两点之间的角度
    angle = np.arctan2(y_14 - y_1, x_14 - x_1)

    # 旋转所有的点，让鱼朝着北方
    x_coords = data[:, 0, :, :]
    y_coords = data[:, 1, :, :]

    rotated_x = x_coords * np.cos(angle)[:, np.newaxis, np.newaxis] - y_coords * np.sin(angle)[:, np.newaxis, np.newaxis]
    rotated_y = x_coords * np.sin(angle)[:, np.newaxis, np.newaxis] + y_coords * np.cos(angle)[:, np.newaxis, np.newaxis]

    rotated_data = data.copy()
    rotated_data[:, 0, :, :] = rotated_x
    rotated_data[:, 1, :, :] = rotated_y

    return rotated_data


def compute_relative_angles(data):
    """
        Compute the angles of each point relative to the 14th point.

        Parameters:
        - data (numpy.ndarray): The input 4-D data array.

        Returns:
        - numpy.ndarray: The output 4-D array with the originnal x, y changed to the angles of each point relative to the 14th point.
    """
    # 获取X, Y坐标
    x_coords = data[:, 0, :, :]
    y_coords = data[:, 1, :, :]

    # 获取第14个点的X, Y坐标
    x_14 = x_coords[:, :, 13]
    y_14 = y_coords[:, :, 13]

    # 计算与第14个点的X, Y坐标差
    delta_x = x_coords - np.expand_dims(x_14, axis=2)
    delta_y = y_coords - np.expand_dims(y_14, axis=2)

    # 使用arctan2计算角度
    angles = np.arctan2(delta_y, delta_x)

    # 调整形状以匹配输出格式
    return angles[:, np.newaxis, :, :]


def tensor_decomposition(data, rank):
    """
    Perform tensor decomposition on the input data and visualize the decomposition results.

    Parameters:
        data (numpy.array): The four-dimensional tensor data to be decomposed.

    Returns:
        factors (numpy.array): The decomposed factors array.
    """
    # Fit an ensemble of models, 4 random replicates / optimization runs per model rank
    ensemble = tt.Ensemble(fit_method="ncp_hals")
    # 修改了ranks=range(1, rank+1)这个范围，对数据张量分别使用秩1到秩(rank+1)进行张量分解
    ensemble.fit(data, ranks=range(1, rank+1), replicates=4)

    fig, axes = plt.subplots(1, 2)
    tt.plot_objective(ensemble, ax=axes[0])  # plot reconstruction error as a function of num components.
    tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
    fig.tight_layout()

    # 里面可以取出来三个矩阵，分别是 swim_bout_factor, time_factor 和 node_factor
    factors = ensemble.factors(rank)[0]

    np.save('tensor_decomposition_factors.npy', factors)

    plt.show()

    return factors


def hierarchical_agglomerative_clustering(swim_bout_factor, n_clusters):
    """
    Performs hierarchical clustering and visualizes the results using a
    dendrogram and a heatmap. Also, compute Davies-Bouldin and Silhouette
    scores for the clustering.
    """

    # 使用AgglomerativeClustering进行聚类
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(swim_bout_factor)

    # 打印一些评价指标
    db_score = davies_bouldin_score(swim_bout_factor, labels)
    s_score = silhouette_score(swim_bout_factor, labels)
    print(f'Davies-Bouldin Score: {db_score}')
    print(f'Silhouette Score: {s_score}')

    # 使用linkage进行聚类并获取排序
    linked = linkage(swim_bout_factor, 'ward')
    order = leaves_list(linked)
    sorted_data = swim_bout_factor[order]

    # 设置图形布局
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 5], wspace=0.05)

    # 绘制dendrogram
    ax0 = fig.add_subplot(gs[0])
    dendro = dendrogram(linked, orientation='left', no_labels=True, ax=ax0, truncate_mode='level', p=5)
    ax0.set_axis_off()

    # 绘制热力图
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(sorted_data, cmap='viridis', yticklabels=False, cbar_kws={'label': 'Weight'}, ax=ax1)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Swim bout number')
    ax1.set_title('Hierarchical clustering of swim bouts')

    plt.show()

    # 返回聚类标签
    return labels


if __name__ == '__main__':

    fishdata = np.load('/Users/lanxinxu/Desktop/INTERN_2023/PoseR/ZebTensor/bouts.npy')
    fishdata = np.squeeze(fishdata, axis=-1)  # 移除尺寸为1的维度，因为只有一条鱼，不需要这个维度
    fishdata_1000 = fishdata[:1000]  # 原始数据太大了，取一部分bouts来看效果

    # 计算每个节点相对于第14个节点的角度
    fishdata_angles = compute_relative_angles(fishdata_1000)

    # 移除尺寸为1的维度，因为角度把xy坐标换成角度以后每个节点只剩一个数了
    fishdata = np.squeeze(fishdata_angles, axis=1)  # fishdata是一个 N * T * V 的数组

    # tensor decomposition
    rank = 10  # 10 components
    tensor_factors = tensor_decomposition(fishdata_1000, rank)
    swim_bout_factor = tensor_factors[0]

    # hierarchical agglomerative clustering 聚类并画出树状图和热力图
    n_clusters = 30  # 希望聚出多少类
    labels = hierarchical_agglomerative_clustering(swim_bout_factor, n_clusters)


