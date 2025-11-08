import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet2_utils import (
    PointNetSetAbstractionMsg, PointNetFeaturePropagation
)

# ==========================================================
#  Channel Attention (SE Block)
# ==========================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, 1)       # [B, C, 1]
        w = F.relu(self.fc1(w))               # [B, C//r, 1]
        w = torch.sigmoid(self.fc2(w))        # [B, C, 1]
        return x * w                          # 通道加权输出


# ==========================================================
#  PointNet++ Dual-Head + Attention
# ==========================================================
class PointNet2DualHeadAttention(nn.Module):
    def __init__(self, coord_dim=3, feature_dim=3):
        """
        coord_dim: 输入坐标维度 (2D=2, 3D=3)
        feature_dim: 额外特征维度 (如颜色/强度等)
        """
        super().__init__()
        self.coord_dim = coord_dim

        # ---------- PointNet++ Backbone ----------
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

        # ---------- Feature Propagation ----------
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # ---------- Attention Blocks ----------
        self.att4 = SEBlock(256)
        self.att3 = SEBlock(256)
        self.att2 = SEBlock(128)
        self.att1 = SEBlock(128)

        # ---------- Shared MLP ----------
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        # ---------- Dual Output Heads ----------
        self.head_path = nn.Conv1d(128, 1, 1)       # 输出路径概率
        self.head_keypoint = nn.Conv1d(128, 1, 1)   # 输出关键点概率


    def forward(self, xyz):
        """
        Input:
            xyz: [B, C_in, N] = 坐标+特征
        Output:
            path_logits, keypoint_logits: [B, N] (raw logits)
        """
        l0_points = xyz
        l0_xyz = xyz[:, :self.coord_dim, :]

        # ------ Hierarchical feature extraction ------
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # ------ Feature Propagation with Attention ------
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points = self.att4(l3_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.att3(l2_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.att2(l1_points)

        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        l0_points = self.att1(l0_points)

        # ------ Shared MLP ------
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)

        # ------ Dual heads ------
        path_logits = self.head_path(x).squeeze(1)       # [B, N]
        keypoint_logits = self.head_keypoint(x).squeeze(1) # [B, N]

        return path_logits, keypoint_logits


# ==========================================================
#  Loss Function: Dual-Head BCE With Logits
# ==========================================================
class DualHeadBCELoss(nn.Module):
    """For soft label supervision (0~1), with optional positive weighting"""
    def __init__(self, pos_weight_path=None, pos_weight_keypoint=None):
        super(DualHeadBCELoss, self).__init__()
        self.loss_path = nn.BCEWithLogitsLoss(pos_weight=pos_weight_path)
        self.loss_keypoint = nn.BCEWithLogitsLoss(pos_weight=pos_weight_keypoint)

    def forward(self, path_logits, keypoint_logits, path_target, keypoint_target):
        """
        Inputs:
            path_logits, keypoint_logits: [B, N] (raw logits)
            path_target, keypoint_target: [B, N] (float in [0,1])
        Outputs:
            total_loss, loss_path, loss_keypoint
        """
        loss_p = self.loss_path(path_logits, path_target)
        loss_k = self.loss_keypoint(keypoint_logits, keypoint_target)
        return loss_p + loss_k, loss_p, loss_k
