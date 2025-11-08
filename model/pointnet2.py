import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import (
    PointNetSetAbstractionMsg, PointNetFeaturePropagation
)

class get_model(nn.Module):
    def __init__(self, coord_dim=3, feature_dim=3):
        super().__init__()
        self.coord_dim = coord_dim

        # ------------------------------
        # PointNet++ Backbone
        # ------------------------------
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32],
                                             coord_dim + feature_dim,
                                             [[16, 16, 32], [32, 32, 64]],
                                             coord_dim)
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32],
                                             32 + 64,
                                             [[64, 64, 128], [64, 96, 128]],
                                             coord_dim)
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32],
                                             128 + 128,
                                             [[128, 196, 256], [128, 196, 256]],
                                             coord_dim)
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32],
                                             256 + 256,
                                             [[256, 256, 512], [256, 384, 512]],
                                             coord_dim)

        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # --- Shared MLP before dual heads ---
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        # --- Dual output heads ---
        self.conv2_path = nn.Conv1d(128, 2, 1)       # 2-class path logits
        self.conv2_keypoint = nn.Conv1d(128, 2, 1)   # 2-class keypoint logits
    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :self.coord_dim, :]

        # PointNet++ 层级特征提取
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # 特征上采样融合
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # --- Shared MLP ---
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))

        # --- Dual heads ---
        path_logits = self.conv2_path(x)         # (B,2,N)
        keypoint_logits = self.conv2_keypoint(x) # (B,2,N)

        return path_logits, keypoint_logits

class get_loss_dualhead(nn.Module):
    def __init__(self):
        super(get_loss_dualhead, self).__init__()

    def forward(self, path_logits, keypoint_logits, path_target, keypoint_target, path_weight=None, keypoint_weight=None):
        """
        path_logits: (B,2,N)
        keypoint_logits: (B,2,N)
        path_target: (B,N)
        keypoint_target: (B,N)
        """
        # print("path_logits shape:", path_logits.shape)
        # print("keypoint_logits shape:", keypoint_logits.shape)
        # print("path_target shape:", path_target.shape)
        # print("keypoint_target shape:", keypoint_target.shape)

        # NLL Loss expects (B,C,N) and target (B,N)
        path_loss = F.nll_loss(F.log_softmax(path_logits, dim=1), path_target, weight=path_weight)
        keypoint_loss = F.nll_loss(F.log_softmax(keypoint_logits, dim=1), keypoint_target, weight=keypoint_weight)

        # print("path_loss:", path_loss.item())
        # print("keypoint_loss:", keypoint_loss.item())

        total_loss = path_loss + keypoint_loss
        return total_loss, path_loss, keypoint_loss
