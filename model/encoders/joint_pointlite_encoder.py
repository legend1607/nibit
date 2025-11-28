import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cae.CNN_3d import Encoder_CNN_3D   # 你之前的 3D CNN
from model.mlp import JointMLPEncoder
from model.encoders.pointnetlite_encoder import PointNetLiteEncoder


class JointPointNetEncoder(nn.Module):
    """
    环境 CNN + 关节 MLP + per-point PointNetLite head
    输出：逐点特征 local_feat (B, N, embed_dim)
    """
    def __init__(
        self,
        joint_in_dim=7,
        joint_feat_dim=64,
        env_latent_dim=60,
        pointnet_embed_dim=256,
        hidden_dims_joint=[128,128],
        pointnet_hidden=[128,256],
        dropout_p=0,
        num_classes=3
    ):
        super().__init__()

        # 1) 环境 CNN → env_feat: (B, env_latent_dim)
        self.env_encoder = Encoder_CNN_3D(dropout_p=dropout_p)

        # 2) 关节 MLP
        self.joint_encoder = JointMLPEncoder(
            joint_in_dim=joint_in_dim,
            joint_feat_dim=joint_feat_dim,
            hidden_dims=hidden_dims_joint
        )


        # 3) PointNetLite for fused features
        self.pointnet = PointNetLiteEncoder(
            in_dim=joint_feat_dim + env_latent_dim,
            embed_dim=pointnet_embed_dim,
            hidden_dims=pointnet_hidden,
            conv_dim=1,
            return_all=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(pointnet_embed_dim, pointnet_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pointnet_embed_dim, num_classes)
        )

    def forward(self, env_voxel, joint_states):
        """
        env_voxel : (B, 1, D, H, W)
        joint_states: (B, N, joint_in_dim)
        """
        B, N, _ = joint_states.shape

        # (1) 环境特征
        env_feat = self.env_encoder(env_voxel)   # (B, env_latent_dim)

        # (2) 关节特征
        joint_feat = self.joint_encoder(joint_states)

        # (3) 融合环境特征
        env_expand = env_feat.unsqueeze(1).expand(B, N, -1)
        fused = torch.cat([joint_feat, env_expand], dim=-1)  # (B, N, joint_feat+env_feat)

        # (4) PointNetLite
        global_feat, local_feat = self.pointnet(fused)

        logits = self.mlp(local_feat)

        return logits, global_feat, local_feat  # local_feat 可用于 per-point 分类
