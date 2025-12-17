import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple, List, Dict, Any


def collate_fn_with_meta_and_pathlabel(batch):
    """
    batch: list of (env_voxel, joint_states, labels, pathlabels, meta)

    只对 env_voxel / joint_states / labels / pathlabels 拼接，
    meta 保持成 list，不去 stack 里面的 path 等可变长内容
    """
    env_voxels, joint_states, labels, pathlabels, metas = zip(*batch)

    env_voxels = torch.stack(env_voxels, dim=0)        # (B, 1, D, H, W)
    joint_states = torch.stack(joint_states, dim=0)   # (B, N, DoF)
    labels = torch.stack(labels, dim=0)               # (B, N)
    pathlabels = torch.stack(pathlabels, dim=0)       # (B, N)

    return env_voxels, joint_states, labels, pathlabels, list(metas)


class ArmPointCloudDataset(Dataset):
    """
    用于 3D CNN + Joint MLP + PointNet 的机械臂规划数据集。

    期望 npz 中至少包含：
        - voxel_grids: (M, D, H, W)
        - pc:          (M, N, DoF)
        - labels:      (M, N)   二分类: 0=collision, 1=free
        - pathlabels:  (M, N)   软标签 float32 in [0,1], 越小 dist 越接近 1
    """

    def __init__(
        self,
        npz_path: str,
        max_points: int = None,
        shuffle_points: bool = False,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        require_pathlabels: bool = True,  # 如果你有旧数据集可设 False 兼容
    ):
        super().__init__()
        self.npz_path = npz_path
        self.max_points = max_points
        self.shuffle_points = shuffle_points
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.require_pathlabels = require_pathlabels

        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        # ---- required fields ----
        self.voxel_grids = data["voxel_grids"]  # (M, D, H, W)
        self.pc = data["pc"]                    # (M, N, DoF)
        self.labels = data["labels"]            # (M, N)

        # ---- new optional/required field ----
        if "pathlabels" in data.files:
            self.pathlabels = data["pathlabels"]  # (M, N) float32
        else:
            self.pathlabels = None
            if self.require_pathlabels:
                raise KeyError(
                    f"`pathlabels` not found in {npz_path}. "
                    f"Please regenerate dataset with soft pathlabels."
                )

        print("self.pc:", self.pc.shape)
        print("self.labels:", self.labels.shape)
        if self.pathlabels is not None:
            print("self.pathlabels:", self.pathlabels.shape)

        # ---- optional meta fields ----
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

    def _maybe_sample_points(self, pc_i, labels_i, pathlabels_i=None):
        """
        对 pc / labels / pathlabels 同步子采样
        """
        N = pc_i.shape[0]
        if self.max_points is None or self.max_points >= N:
            idx = np.arange(N)
        else:
            idx = np.random.choice(N, self.max_points, replace=False)

        if self.shuffle_points:
            np.random.shuffle(idx)

        pc_o = pc_i[idx]
        labels_o = labels_i[idx]
        if pathlabels_i is None:
            return pc_o, labels_o, None
        else:
            return pc_o, labels_o, pathlabels_i[idx]

    def __getitem__(self, idx):
        voxel = self.voxel_grids[idx]
        pc_i = self.pc[idx]
        labels_i = self.labels[idx]

        if self.pathlabels is not None:
            pathlabels_i = self.pathlabels[idx]
        else:
            pathlabels_i = None

        pc_i, labels_i, pathlabels_i = self._maybe_sample_points(pc_i, labels_i, pathlabels_i)

        # ---- tensors ----
        env_voxel = torch.from_numpy(voxel).clone().unsqueeze(0).to(self.dtype)   # (1, D, H, W)
        joint_states = torch.from_numpy(pc_i).clone().to(self.dtype)              # (N, DoF)
        labels_t = torch.from_numpy(labels_i).clone().long()                      # (N,)

        if pathlabels_i is not None:
            pathlabels_t = torch.from_numpy(pathlabels_i).clone().to(self.dtype)  # (N,)
        else:
            # 兼容旧数据：全 0
            pathlabels_t = torch.zeros_like(labels_t, dtype=self.dtype)

        # ---- meta stays on CPU ----
        meta: Dict[str, Any] = {}
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
            meta["obstacles"] = self.obstacles[idx]

        # return 5-tuple
        return env_voxel, joint_states, labels_t, pathlabels_t, meta
