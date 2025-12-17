import torch
import torch.nn as nn
from model.mlp import JointMLPEncoder
from model.encoders.pointnetlite_encoder import AttentionPointNet
from model.cae.CNN_3d import Encoder_CNN_3D

class JointPointNetEncoder(nn.Module):
    def __init__(
        self,
        joint_in_dim=4,
        joint_feat_dim=64,
        env_latent_dim=64,     # Encoder_CNN_3D token_dim
        pointnet_embed_dim=128,
        hidden_dims_joint=[128, 128],
        pointnet_hidden=[128, 256],
        # attention
        cross_num_heads: int = 4,
        self_attn_heads: int = 4,
        self_attn_layers: int = 1,
        self_attn_dropout: float = 0.1,
        self_attn_ff_multiplier: float = 2.0,
        use_attn_pool: bool = True,
        # dropout
        point_mlp_dropout: float = 0.1,
        point_feat_dropout: float = 0.1,
        cls_dropout: float = 0.1,
        path_dropout: float = 0.1,
        cross_dropout: float = 0.1,
    ):
        super().__init__()

        # 1) env encoder: (B,64) + (B,4,64)
        self.env_encoder = Encoder_CNN_3D(token_dim=env_latent_dim, token_grid=(2, 2, 1))

        # 2) joint encoder: (B,N,4)->(B,N,64)
        self.joint_encoder = JointMLPEncoder(
            joint_in_dim=joint_in_dim,
            joint_feat_dim=joint_feat_dim,
            hidden_dims=hidden_dims_joint
        )

        # 3) 如果 env_latent_dim != joint_feat_dim，用投影对齐（更鲁棒）
        self.env_align = nn.Sequential(
            nn.Linear(env_latent_dim, joint_feat_dim),
            nn.LayerNorm(joint_feat_dim)
        )


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=joint_feat_dim,
            num_heads=cross_num_heads,
            dropout=cross_dropout,
            batch_first=True
        )

        # Pre-Norm
        self.cross_ln_q  = nn.LayerNorm(joint_feat_dim)
        self.cross_ln_ff = nn.LayerNorm(joint_feat_dim)
        self.cross_drop  = nn.Dropout(cross_dropout)

        self.cross_ff = nn.Sequential(
            nn.Linear(joint_feat_dim, joint_feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(joint_feat_dim * 2, joint_feat_dim),
        )

        # 4) pointnet: concat joint(64) + env_global(64) => 128
        self.pointnet = AttentionPointNet(
            in_dim=joint_feat_dim + joint_feat_dim,  # 64 + 64
            embed_dim=pointnet_embed_dim,
            hidden_dims=pointnet_hidden,
            num_heads=self_attn_heads,
            num_layers=self_attn_layers,
            attn_dropout=self_attn_dropout,
            ff_multiplier=self_attn_ff_multiplier,
            use_attn_pool=use_attn_pool,
            mlp_dropout=point_mlp_dropout,
            feat_dropout=point_feat_dropout,
        )

        # 5) heads
        self.shared_head_fc = nn.Linear(pointnet_embed_dim * 2, pointnet_embed_dim)
        self.shared_head_act = nn.ReLU(inplace=True)

        self.cls_dropout = nn.Dropout(cls_dropout)
        self.path_dropout = nn.Dropout(path_dropout)

        self.cls_out = nn.Linear(pointnet_embed_dim, 1)
        self.path_out = nn.Linear(pointnet_embed_dim, 1)

    def encode_env(self, env_voxel: torch.Tensor):
        # 返回 (env_global, env_tokens)
        return self.env_encoder(env_voxel, return_tokens=True)

    def forward_with_env_feat(self, env_feat, joint_states: torch.Tensor):
        """
        env_feat 支持两种形式：
          1) env_feat = env_global: Tensor(B, C)   -> 自动退化成 1-token cross-attn
          2) env_feat = (env_global, env_tokens)   -> 4-token cross-attn（推荐）
        joint_states: (B, N, joint_in_dim)
        """
        B, N, _ = joint_states.shape

        # --- 解析 env_feat ---
        if isinstance(env_feat, (tuple, list)):
            env_global, env_tokens = env_feat   # (B,C), (B,T,C)
        else:
            env_global, env_tokens = env_feat, None  # 只有 global

        # 兼容：env_global batch=1 但 joint_states batch=B
        if env_global.dim() == 2 and env_global.size(0) == 1 and B > 1:
            env_global = env_global.expand(B, -1)
            if env_tokens is not None and env_tokens.size(0) == 1:
                env_tokens = env_tokens.expand(B, -1, -1)

        # 对齐到 joint_feat_dim（保证 MHA 输入维度一致）
        env_global = self.env_align(env_global)  # (B,64)

        if env_tokens is None:
            # 退化：只有 1 token（仍可跑，但注意力“选择性”会很弱）
            env_tokens = env_global.unsqueeze(1)        # (B,1,64)
        else:
            env_tokens = self.env_align(env_tokens)     # (B,T,64)

        # --- joint ---
        joint_feat = self.joint_encoder(joint_states)   # (B,N,64)

        # --- Cross-Attn (Pre-Norm) ---
        q = self.cross_ln_q(joint_feat)
        cross_out, _ = self.cross_attn(q, env_tokens, env_tokens)  # (B,N,64)
        joint_feat = joint_feat + self.cross_drop(cross_out)

        # --- FFN (Pre-Norm) ---
        ff_in = self.cross_ln_ff(joint_feat)
        joint_feat = joint_feat + self.cross_drop(self.cross_ff(ff_in))

        # --- Fuse for PointNet ---
        env_expand = env_global.unsqueeze(1).expand(B, N, -1)      # (B,N,64)
        fused = torch.cat([joint_feat, env_expand], dim=-1)        # (B,N,128)

        global_feat, local_feat = self.pointnet(fused)

        global_expand = global_feat.unsqueeze(1).expand(B, N, -1)
        feat = torch.cat([local_feat, global_expand], dim=-1)
        shared = self.shared_head_act(self.shared_head_fc(feat))   # (B,N,E)=128

        logits = self.cls_out(self.cls_dropout(shared)).squeeze(-1)        # (B,N)
        pathlogits = self.path_out(self.path_dropout(shared)).squeeze(-1)  # (B,N)

        return logits, pathlogits, local_feat

    def forward(self, env_voxel: torch.Tensor, joint_states: torch.Tensor):
        env_feat = self.encode_env(env_voxel)   # (env_global, env_tokens)
        return self.forward_with_env_feat(env_feat, joint_states)
