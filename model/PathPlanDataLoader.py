# pointnet_pointnet2tf/PathPlanDataLoader.py
import numpy as np
from torch.utils.data import Dataset
from model.pointnet2_utils import pc_normalize

class PathPlanDataset(Dataset):
    def __init__(self, env_type, dataset_filepath):
        data = np.load(dataset_filepath, allow_pickle=True)

        # 必要字段
        self.pc = data["pc"].astype(np.float32)          # (B, N, d)
        self.start_mask = data["start"].astype(np.float32)  # (B, N)
        self.goal_mask = data["goal"].astype(np.float32)
        self.free_mask = data["free"].astype(np.float32)

        # 路径 mask 根据场景类型选择
        self.path_mask = data["path"].astype(np.float32)

        if "keypoint" in data:
            self.keypoint = data["keypoint"].astype(np.float32)
        else:
            # 如果旧数据集没这个字段，则用全零占位
            self.keypoint = np.zeros_like(self.path_mask)

        self.ID = data["token"]

        # 自动维度信息
        self.d = self.pc.shape[2]      # 点云维度
        self.n_points = self.pc.shape[1]
        # print(f"[PathPlanDatasetTF] Loaded point cloud with dimension = {self.d}, n_points = {self.n_points}")

        # 类别权重 (平衡 start/goal/path)
        labelweights, _ = np.histogram(self.path_mask, bins=2, range=(0, 2))
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / (labelweights + 1e-6), 1 / 3.0)
        # print(f"[PathPlanDatasetTF] labelweights = {self.labelweights}")

    def __len__(self):
        return len(self.pc)

    def __getitem__(self, index):
        pc_pos_raw = self.pc[index]   # (N, d)
        pc_pos, centroid, scale = pc_normalize(pc_pos_raw, return_stats=True)

        # 特征拼接：start/goal/free mask
        pc_features = np.stack(
            (self.start_mask[index], self.goal_mask[index], self.free_mask[index]),
            axis=-1,
        )  # (N, 3)

        pc_labels = self.path_mask[index]        # (N,)
        keypoint_labels = self.keypoint[index]       # (N,)

        return (
            pc_pos_raw,       # 原始点云
            pc_pos,           # 归一化点云
            pc_features,      # 起点/终点/自由空间特征
            pc_labels,        # 路径标签
            keypoint_labels,   # 关键点标签
            self.ID[index],    # 环境ID
        )
