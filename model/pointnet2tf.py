# pointnet_pointnet2/models/pointnet2tf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation
)

# ----------------------------
# Global Transformer Module
# ----------------------------
class PointTransformerModule(nn.Module):
    def __init__(self, feature_dim=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        x_out = self.transformer(x)
        return self.mlp(x_out) + x  # residual connection


# ----------------------------
# Learnable Positional Encoding
# ----------------------------
class PositionalEncodingLearned(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, coords):
        coords = coords.permute(0, 2, 1)  # (B, N, in_dim)
        return self.fc(coords)


# ----------------------------
# PointNet++ + Transformer Model
# ----------------------------
class get_model(nn.Module):
    def __init__(self, num_classes=1, coord_dim=3, feature_dim=3, use_direction=True):
        super().__init__()
        self.coord_dim = coord_dim
        self.use_direction = use_direction

        # PointNet++ backbone
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32],
            in_channel=coord_dim + feature_dim,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            coord_dim=coord_dim
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32],
            in_channel=32 + 64,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            coord_dim=coord_dim
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=64, radius_list=[0.2, 0.4], nsample_list=[16, 32],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
            coord_dim=coord_dim
        )
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=16, radius_list=[0.4, 0.8], nsample_list=[16, 32],
            in_channel=256 + 256,
            mlp_list=[[256, 256, 512], [256, 384, 512]],
            coord_dim=coord_dim
        )

        # Feature propagation
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128])

        # Global Transformer
        self.pos_enc = PositionalEncodingLearned(in_dim=coord_dim, out_dim=128)
        self.transformer_module = PointTransformerModule(feature_dim=128)

        # Path head
        self.path_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
        )

        # Keypoint head
        self.keypoint_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1)
        )

        # Optional direction head
        if use_direction:
            self.direction_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, coord_dim, 1)
            )

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :self.coord_dim, :]

        # PointNet++ backbone
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # Global context
        pos_emb = self.pos_enc(l0_xyz)
        x = l0_points.permute(0, 2, 1) + pos_emb
        x = self.transformer_module(x)
        x = x.permute(0, 2, 1)

        # Outputs
        path_logits = self.path_head(x)
        keypoint_logits = self.keypoint_head(x)

        if self.use_direction:
            direction = F.normalize(self.direction_head(x), dim=1)
            return path_logits, keypoint_logits, direction
        else:
            return path_logits, keypoint_logits, None


# ----------------------------
# Loss Function
# ----------------------------
class get_loss(nn.Module):
    def __init__(self, w_path=1.0, w_keypoint=2.0, w_direction=3, use_direction=True):
        super().__init__()
        self.w_path = w_path
        self.w_keypoint = w_keypoint
        self.w_direction = w_direction
        self.use_direction = use_direction
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, path_pred, keypoint_pred, direction_pred,
                path_label, keypoint_label, direction_label=None):
        path_loss = self.bce(path_pred.squeeze(1), path_label)
        keypoint_loss = self.bce(keypoint_pred.squeeze(1), keypoint_label)

        if self.use_direction and direction_pred is not None and direction_label is not None:
            direction_pred = F.normalize(direction_pred, dim=1)
            direction_label = F.normalize(direction_label.permute(0, 2, 1), dim=1)
            valid_mask = (direction_label.abs().sum(dim=1) > 1e-3).float()
            cos_sim = F.cosine_similarity(direction_pred, direction_label, dim=1)
            direction_loss = ((1 - cos_sim) * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            direction_loss = torch.tensor(0.0, device=path_pred.device)

        total_loss = (
            self.w_path * path_loss +
            self.w_keypoint * keypoint_loss +
            self.w_direction * direction_loss
        )

        return total_loss, path_loss, keypoint_loss, direction_loss