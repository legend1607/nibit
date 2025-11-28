# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union

def collate_fn_with_meta(batch):
    """
    batch: list of (env_voxel, joint_states, labels, meta)
    只对前三个做默认拼接，meta 保持成 list，不去 stack 里面的 path 等可变长内容
    """
    env_voxels, joint_states, labels, metas = zip(*batch)  # 解包

    env_voxels = torch.stack(env_voxels, dim=0)   # (B, 1, D, H, W)
    joint_states = torch.stack(joint_states, dim=0)  # (B, N, DoF)
    labels = torch.stack(labels, dim=0)           # (B, N)

    # metas 直接作为 tuple/list 返回，不做进一步拼接
    return env_voxels, joint_states, labels, list(metas)

class ArmPointCloudDataset(Dataset):
    """
    用于 3D CNN + Joint MLP + PointNet 的机械臂规划数据集。

    期望 npz 中至少包含：
        - voxel_grids: (M, D, H, W)
        - pc:          (M, N, DoF)
        - labels:      (M, N)
    """

    def __init__(
        self,
        npz_path: str,
        max_points: int = None,
        shuffle_points: bool = False,
        device: Union[torch.device, str, None] = None,  # 修复这里
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
        print("self.pc,",self.pc.shape)
        # optional fields
        self.starts = data["starts"] if "starts" in data.files else None
        self.goals = data["goals"] if "goals" in data.files else None
        self.env_ranges = data["env_ranges"] if "env_ranges" in data.files else None
        self.pose_ranges = data["pose_ranges"] if "pose_ranges" in data.files else None
        self.paths = data["paths"] if "paths" in data.files else None
        self.obstacles = data["obstacles"] if "obstacles" in data.files else None

        self.num_envs, self.num_points, self.dof = self.pc.shape

        if self.max_points is not None and self.max_points > self.num_points:
            raise ValueError(
                f"max_points={self.max_points} > N={self.num_points}，"
                f"目前实现仅支持 N 内随机采样，不做 zero-padding。"
            )

    def __len__(self):
        return self.num_envs

    def _maybe_sample_points(self, pc_i, labels_i):
        N = pc_i.shape[0]
        if self.max_points is None or self.max_points >= N:
            idx = np.arange(N)
        else:
            idx = np.random.choice(N, self.max_points, replace=False)

        if self.shuffle_points:
            np.random.shuffle(idx)

        return pc_i[idx], labels_i[idx]

    def __getitem__(self, idx):
        voxel = self.voxel_grids[idx]
        pc_i = self.pc[idx]
        labels_i = self.labels[idx]

        pc_i, labels_i = self._maybe_sample_points(pc_i, labels_i)

        # 必须 clone()，必须是 CPU
        env_voxel = torch.from_numpy(voxel).clone().unsqueeze(0).to(self.dtype)
        joint_states = torch.from_numpy(pc_i).clone().to(self.dtype)
        labels_t = torch.from_numpy(labels_i).clone().long()

        # meta 保持 CPU + 不要在 DataLoader 里堆叠
        meta = {}
        if self.starts is not None:
            meta["start"] = torch.from_numpy(self.starts[idx]).clone().to(self.dtype)
        if self.goals is not None:
            meta["goal"] = torch.from_numpy(self.goals[idx]).clone().to(self.dtype)
        if self.env_ranges is not None:
            meta["env_range"] = torch.from_numpy(self.env_ranges[idx]).clone().to(self.dtype)
        if self.pose_ranges is not None:
            meta["pose_range"] = torch.from_numpy(self.pose_ranges[idx]).clone().to(self.dtype)
        if self.paths is not None:
            meta["path"] = torch.from_numpy(self.paths[idx]).clone().to(self.dtype)
        if self.obstacles is not None:
            meta["obstacles"] = self.obstacles[idx]  # 不转 tensor，保持原样

        return env_voxel, joint_states, labels_t, meta
