import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import numpy as np

dataset_path = "data/random_2d/val.npz"  # 替换成你的 npz 文件路径
data = np.load(dataset_path, allow_pickle=True)

print("包含的字段：", list(data.keys()))

# ===============================
# 🔹 基本数据
# ===============================
keypoints = data["keypoint"]
path = data["path"]
pc = data["pc"]

print("\n--- 数据形状信息 ---")
print("pc shape:", pc.shape)
print("keypoint shape:", keypoints.shape)
print("path shape:", path.shape)

# ===============================
# 🔹 keypoint 统计
# ===============================
print("\n--- keypoint 统计 ---")
keypoints_nonzero_counts = np.sum(keypoints > 0, axis=1)
# print("每个样本非零关键点数:", keypoints_nonzero_counts)
print("总体非零关键点总数:", np.sum(keypoints_nonzero_counts))
print("最大关键点数:", np.max(keypoints_nonzero_counts), 
      "最小关键点数:", np.min(keypoints_nonzero_counts))

no_keypoint_samples = np.sum(keypoints_nonzero_counts == 0)
print("关键点全零样本数:", no_keypoint_samples)
print("关键点全零样本占比: {:.2%}".format(no_keypoint_samples / len(keypoints)))

print("\nkeypoint dtype:", keypoints.dtype)
print("keypoint min:", np.min(keypoints), "max:", np.max(keypoints), "mean:", np.mean(keypoints))
print("keypoint unique (前20):", np.unique(keypoints)[:20])

num_keypoints_above_05 = np.sum(keypoints > 0.5)
print("keypoint > 0.5 的总数量:", num_keypoints_above_05)
num_keypoints_per_sample = np.sum(keypoints > 0.5, axis=1)
print("每个样本 keypoint > 0.5 数量最大:", np.max(num_keypoints_per_sample))
print("每个样本 keypoint > 0.5 数量最小:", np.min(num_keypoints_per_sample))
print("每个样本 keypoint > 0.5 数量平均:", np.mean(num_keypoints_per_sample))

# ===============================
# 🔹 path 统计
# ===============================
print("\n--- path 统计 ---")

# 计算每个样本中 path 非零点数量
path_nonzero_counts = np.sum(path > 0, axis=1)
# print("每个样本非零 path 数量:", path_nonzero_counts)

print("总体非零 path 点总数:", np.sum(path_nonzero_counts))
print("最大 path 数量:", np.max(path_nonzero_counts),
      "最小 path 数量:", np.min(path_nonzero_counts))
print("平均每个样本 path 数量:", np.mean(path_nonzero_counts))

# 检查没有路径的样本
no_path_samples = np.sum(path_nonzero_counts == 0)
print("path 全零样本数:", no_path_samples)
print("path 全零样本占比: {:.2%}".format(no_path_samples / len(path)))

# 路径取值范围与分布
print("\npath dtype:", path.dtype)
print("path min:", np.min(path), "max:", np.max(path), "mean:", np.mean(path))
print("path unique (前20):", np.unique(path)[:20])

# 如果 path 是连续值，可以检查阈值分布
num_path_above_05 = np.sum(path > 0.5)
print("path > 0.5 的总数量:", num_path_above_05)

num_path_per_sample = np.sum(path > 0.5, axis=1)
print("每个样本 path > 0.5 数量最大:", np.max(num_path_per_sample))
print("每个样本 path > 0.5 数量最小:", np.min(num_path_per_sample))
print("每个样本 path > 0.5 数量平均:", np.mean(num_path_per_sample))

# ===============================
# 🔹 综合信息
# ===============================
print("\n--- 综合样本统计 ---")
total_samples = len(keypoints)
print(f"样本总数: {total_samples}")
print(f"无关键点样本比例: {no_keypoint_samples / total_samples:.2%}")
print(f"无路径样本比例: {no_path_samples / total_samples:.2%}")
print(f"平均每样本关键点数: {np.mean(keypoints_nonzero_counts):.2f}")
print(f"平均每样本路径点数: {np.mean(path_nonzero_counts):.2f}")
# ===============================
# 读取 npz 数据集
# ===============================
dataset_dir = "data/random_2d"  # 替换为你的数据集路径
mode = "train"  # 或 "val"/"test"
npz_path = join(dataset_dir, f"{mode}.npz")

data = np.load(npz_path, allow_pickle=True)

tokens = data["token"]
pcs = data["pc"]
starts = data["start"]
goals = data["goal"]
frees = data["free"]
paths = data["path"]
keypoints = data["keypoint"]

print(f"Loaded {len(tokens)} samples from {npz_path}")

# ===============================
# 可视化单个样本
# ===============================
def visualize_sample(idx):
    pc = pcs[idx]
    start_mask = starts[idx].astype(bool)
    goal_mask = goals[idx].astype(bool)
    free_mask = frees[idx].astype(bool)
    path_label = paths[idx]
    keypoint_label = keypoints[idx]
    
    plt.figure(figsize=(6, 6))
    
    # 绘制点云（灰色背景）
    plt.scatter(pc[:, 0], pc[:, 1], c='lightgray', s=5, label='Point cloud')
    
    # 绘制起点和终点
    plt.scatter(pc[start_mask, 0], pc[start_mask, 1],
                c='green', s=80, marker='*', edgecolors='k', label='Start')
    plt.scatter(pc[goal_mask, 0], pc[goal_mask, 1],
                c='magenta', s=80, marker='*', edgecolors='k', label='Goal')
    
    # 仅绘制 path > 0.5 的点
    path_mask = path_label > 0.5
    if np.any(path_mask):
        plt.scatter(pc[path_mask, 0], pc[path_mask, 1],
                    c='red', s=25, alpha=0.8, label='Path > 0.5')
    
    # 仅绘制 keypoint > 0.5 的点
    keypoint_mask = keypoint_label > 0.5
    if np.any(keypoint_mask):
        plt.scatter(pc[keypoint_mask, 0], pc[keypoint_mask, 1],
                    c='orange', s=30, marker='x', alpha=0.9, label='Keypoint > 0.5')
    
    plt.title(tokens[idx])
    plt.legend(loc='upper right', fontsize=8)
    plt.axis('equal')
    plt.show()

# ===============================
# 可视化所有样本
# ===============================
for i in range(len(tokens)):
    visualize_sample(i)
