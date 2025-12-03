import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cae.CNN_3d import Encoder_CNN_3D   # 3D CNN
from model.mlp import JointMLPEncoder
from model.encoders.pointnetlite_encoder import AttentionPointNet

class JointPointNetEncoder(nn.Module):
    """
    环境 CNN + 关节 MLP + AttentionPointNet head
    输出：逐点特征 local_feat (B, N, embed_dim)
    """
    def __init__(
        self,
        joint_in_dim=7,
        joint_feat_dim=48,
        env_latent_dim=60,
        pointnet_embed_dim=128,
        hidden_dims_joint=[128, 128],
        pointnet_hidden=[128, 256],
        num_classes=3,
        # attention 相关超参数
        cross_num_heads: int = 4,
        self_attn_heads: int = 4,
        self_attn_layers: int = 1,
        self_attn_dropout: float = 0.1,
        self_attn_ff_multiplier: float = 2.0,
        use_attn_pool: bool = True,
        # 🔥 新增：Dropout 超参数
        point_mlp_dropout: float = 0.1,   # AttentionPointNet 里的 per-point MLP
        point_feat_dropout: float = 0.1,  # self-attn 之后的特征 dropout
        cls_dropout: float = 0.1,         # 最后的分类 head
    ):
        super().__init__()

        # 1) 环境 CNN → env_feat: (B, env_latent_dim)
        self.env_encoder = Encoder_CNN_3D()

        # 2) 关节 MLP  （如要在这里加 dropout，需要在 JointMLPEncoder 里改）
        self.joint_encoder = JointMLPEncoder(
            joint_in_dim=joint_in_dim,
            joint_feat_dim=joint_feat_dim,
            hidden_dims=hidden_dims_joint
        )

        # 3) Cross-Attention: 用环境特征调制关节特征
        self.env_proj = nn.Linear(env_latent_dim, joint_feat_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_feat_dim,
            num_heads=cross_num_heads,
            dropout=0.1,          # 这里保持原样，MultiheadAttention 内部自带 dropout
            batch_first=True
        )
        self.cross_ln1 = nn.LayerNorm(joint_feat_dim)
        self.cross_ff = nn.Sequential(
            nn.Linear(joint_feat_dim, joint_feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(joint_feat_dim * 2, joint_feat_dim),
        )
        self.cross_ln2 = nn.LayerNorm(joint_feat_dim)

        # 4) AttentionPointNet for fused features (self-attention on points)
        #    👉 把 point_mlp_dropout / point_feat_dropout 传进去
        self.pointnet = AttentionPointNet(
            in_dim=joint_feat_dim + env_latent_dim,
            embed_dim=pointnet_embed_dim,
            hidden_dims=pointnet_hidden,
            num_heads=self_attn_heads,
            num_layers=self_attn_layers,
            attn_dropout=self_attn_dropout,
            ff_multiplier=self_attn_ff_multiplier,
            use_attn_pool=use_attn_pool,
            mlp_dropout=point_mlp_dropout,     # 🔥 新增
            feat_dropout=point_feat_dropout,   # 🔥 新增
        )

        # 5) per-point 分类 head  👉 这里加一个 Dropout
        self.mlp = nn.Sequential(
            nn.Linear(pointnet_embed_dim, pointnet_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),           # 🔥 新增
            nn.Linear(pointnet_embed_dim, num_classes)
        )

    # ------------------------------------------------------------------
    # encode_env / forward_with_env_feat / forward 和你原来完全一样
    # ------------------------------------------------------------------
    def encode_env(self, env_voxel: torch.Tensor) -> torch.Tensor:
        env_feat = self.env_encoder(env_voxel)
        return env_feat

    def forward_with_env_feat(
        self,
        env_feat: torch.Tensor,
        joint_states: torch.Tensor
    ):
        B, N, _ = joint_states.shape

        if env_feat.dim() == 2 and env_feat.size(0) == 1 and B > 1:
            env_feat = env_feat.expand(B, -1)

        joint_feat = self.joint_encoder(joint_states)       # (B, N, F_joint)

        env_token = self.env_proj(env_feat).unsqueeze(1)    # (B, 1, F_joint)

        cross_out, _ = self.cross_attn(
            query=joint_feat,
            key=env_token,
            value=env_token
        )
        joint_feat = self.cross_ln1(joint_feat + cross_out)

        ff_out = self.cross_ff(joint_feat)
        joint_feat = self.cross_ln2(joint_feat + ff_out)    # (B, N, F_joint)

        env_expand = env_feat.unsqueeze(1).expand(B, N, -1) # (B, N, F_env)
        fused = torch.cat([joint_feat, env_expand], dim=-1) # (B, N, F_joint+F_env)

        global_feat, local_feat = self.pointnet(fused)      # (B, E), (B, N, E)

        logits = self.mlp(local_feat)                       # (B, N, num_classes)

        return logits, global_feat, local_feat

    def forward(self, env_voxel: torch.Tensor, joint_states: torch.Tensor):
        env_feat = self.encode_env(env_voxel)
        return self.forward_with_env_feat(env_feat, joint_states)
