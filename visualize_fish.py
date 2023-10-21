import numpy as np
import matplotlib.pyplot as plt


# ... specify a numpy array holding the tensor you wish to fit
data = np.load('/Users/lanxinxu/Desktop/INTERN_2023/PoseR/ZebTensor/bouts.npy')

# 使用 squeeze 函数移除尺寸为1的维度，因为只有一条鱼，不需要这个维度
data = np.squeeze(data, axis=-1)

# 取出想要看的回合的所有帧
all_frames = data[0, :, :, :]

# 为每一帧创建一个图形
for i, frame in enumerate(all_frames.transpose(1, 0, 2)):
    x_coords = frame[0]
    y_coords = frame[1]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='blue', marker='o')

    # 设置坐标轴的范围和标题
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Frame {i + 1} of the first round')

    # 保存图形
    plt.savefig(f'frame_1.{i + 1}.png')
    plt.close()
