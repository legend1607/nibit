# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ArmPointCloudDataset(Dataset):
    """
    用于 3D CNN + Joint MLP + PointNet 的机械臂规划数据集。

    期望 npz 中至少包含：
        - voxel_grids: (M, D, H, W)
        - pc:          (M, N, DoF)
        - labels:      (M, N)
    其他字段（可选）：
        - starts:      (M, DoF)
        - goals:       (M, DoF)
        - env_ranges:  (M, 3, 2)
        - pose_ranges: (M, DoF, 2)
        - paths:       object 数组，每个元素是 (L_i, DoF)
        - obstacles:   object 数组，列表，每个元素为 [(type, size, pos), ...]
    """

    def __init__(
        self,
        npz_path: str,
        max_points: int = None,
        shuffle_points: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.npz_path = npz_path
        self.max_points = max_points
        self.shuffle_points = shuffle_points
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype

        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # 重要：obstacles/path 之类是 object，需要 allow_pickle=True
        data = np.load(npz_path, allow_pickle=True)

        self.voxel_grids = data["voxel_grids"]      # (M, D, H, W)
        self.pc = data["pc"]                        # (M, N, DoF)
        self.labels = data["labels"]                # (M, N)

        # 可选字段：使用 get 的方式，缺了也不报错
        self.starts = data["starts"] if "starts" in data.files else None
        self.goals = data["goals"] if "goals" in data.files else None
        self.env_ranges = data["env_ranges"] if "env_ranges" in data.files else None
        self.pose_ranges = data["pose_ranges"] if "pose_ranges" in data.files else None
        self.paths = data["paths"] if "paths" in data.files else None
        self.obstacles = data["obstacles"] if "obstacles" in data.files else None

        self.num_envs, self.num_points, self.dof = self.pc.shape

        # 如果指定了 max_points，检查一下合理性（只支持 截断/下采样，不自动填充）
        if self.max_points is not None and self.max_points > self.num_points:
            raise ValueError(
                f"max_points={self.max_points} > N={self.num_points}，"
                f"目前实现仅支持 N 内随机采样，不做 zero-padding。"
            )

    def __len__(self):
        return self.num_envs

    def _maybe_sample_points(self, pc_i, labels_i):
        """
        根据 max_points 对点云进行随机子采样。
        pc_i:     (N, DoF)
        labels_i: (N,)
        """
        N = pc_i.shape[0]
        if self.max_points is None or self.max_points >= N:
            # 不裁剪
            idx = np.arange(N)
        else:
            # 随机下采样 max_points 个点
            idx = np.random.choice(N, self.max_points, replace=False)

        if self.shuffle_points:
            np.random.shuffle(idx)

        return pc_i[idx], labels_i[idx]

    def __getitem__(self, idx):
        """
        返回：
            env_voxel:   (1, D, H, W)  float32
            joint_states:(N, DoF)     float32
            labels:      (N,)         long
            以及附加信息（字典形式），便于调试或可视化
        """
        voxel = self.voxel_grids[idx]     # (D, H, W)
        pc_i = self.pc[idx]               # (N, DoF)
        labels_i = self.labels[idx]       # (N,)

        # 可选：随机子采样/打乱点
        pc_i, labels_i = self._maybe_sample_points(pc_i, labels_i)

        # 转 tensor
        env_voxel = torch.from_numpy(voxel).unsqueeze(0)  # (1, D, H, W)
        joint_states = torch.from_numpy(pc_i)
        labels_t = torch.from_numpy(labels_i)

        env_voxel = env_voxel.to(self.dtype)
        joint_states = joint_states.to(self.dtype)
        labels_t = labels_t.long()

        if self.device is not None:
            env_voxel = env_voxel.to(self.device, non_blocking=True)
            joint_states = joint_states.to(self.device, non_blocking=True)
            labels_t = labels_t.to(self.device, non_blocking=True)

        # 附加信息（不一定都存在）
        meta = {}
        if self.starts is not None:
            meta["start"] = torch.from_numpy(self.starts[idx]).to(self.dtype)
        if self.goals is not None:
            meta["goal"] = torch.from_numpy(self.goals[idx]).to(self.dtype)
        if self.env_ranges is not None:
            meta["env_range"] = torch.from_numpy(self.env_ranges[idx]).to(self.dtype)
        if self.pose_ranges is not None:
            meta["pose_range"] = torch.from_numpy(self.pose_ranges[idx]).to(self.dtype)
        if self.paths is not None:
            # paths[idx] 是一个 (L_i, DoF) 的 numpy 数组
            path_arr = self.paths[idx]
            meta["path"] = torch.from_numpy(path_arr).to(self.dtype)
        if self.obstacles is not None:
            # 障碍保持为 Python list，通常用于可视化
            meta["obstacles"] = self.obstacles[idx]

        # 统一返回格式：env_voxel, joint_states, labels, meta
        return env_voxel, joint_states, labels_t, meta
