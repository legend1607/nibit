import torch
import torch.nn as nn


class FastSharedMLP(nn.Module):
    """
    与 build_shared_mlp 等价，但使用 Linear + reshape，
    在 conv_dim=1 （PointNet 标准）下速度更快。
    """
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):       # x: (B, N, C_in)
        B, N, C = x.shape
        x = self.mlp(x)         # (B, N, C_out)
        return x


class PointNetLiteEncoder(nn.Module):
    """
    原始 Lite 版 PointNet 编码器：
    - 移除多余 permute
    - Linear 替代 Conv1d(kernel=1)
    - 用 torch.amax 加速 max pooling
    """
    def __init__(self, in_dim=3, embed_dim=1024, hidden_dims=[64, 128],
                 conv_dim=1, return_all=False):
        super().__init__()
        self.grouped = (conv_dim != 1)
        self.return_all = return_all

        # 构建 MLP：等价替代 build_shared_mlp
        mlp_dims = [in_dim] + hidden_dims + [embed_dim]

        if not self.grouped:
            # 标准 PointNet flow (B, N, C)
            self.mlp = FastSharedMLP(mlp_dims)
        else:
            # grouped 情况保持一致（仍使用 conv）
            from model.ECSP.modules.builders import build_shared_mlp
            self.mlp = build_shared_mlp(mlp_dims, conv_dim=2)

    def forward(self, x):
        """
        x:
            - (B, N, C)   if not grouped
            - (B, G, S, C) if grouped
        """
        if not self.grouped:
            # 保持 (B, N, C)，Linear 直接处理
            feat = self.mlp(x)                      # (B, N, embed_dim)
            global_feat = torch.amax(feat, dim=1)   # (B, embed_dim)

            if self.return_all:
                return global_feat, feat            # (B, embed_dim), (B, N, embed_dim)
            return global_feat

        else:
            # grouped 情况：保持你原来的逻辑
            x = x.permute(0, 3, 2, 1)
            feat = self.mlp(x)
            global_feat = torch.amax(feat, dim=2)

            global_feat = global_feat.permute(0, 2, 1)

            if self.return_all:
                local_feat = feat.permute(0, 3, 2, 1)
                return global_feat, local_feat
            return global_feat

# ----------------------------------------------------------------------
# AttentionPointNet：用于替换 JointPointNetEncoder 里的 PointNetLiteEncoder
#   - per-point MLP 提升维度
#   - Self-Attention 沿 N 维建模点与点之间的关系
#   - (可选) Attention Pooling / Max Pooling 获取 global feature
# ----------------------------------------------------------------------
class AttentionPointNet(nn.Module):
    """
    输入:  x (B, N, C_in)
    输出:  global_feat (B, embed_dim)
          local_feat  (B, N, embed_dim)
    """
    def __init__(
        self,
        in_dim: int = 3,
        embed_dim: int = 256,
        hidden_dims = [128, 256],
        num_heads: int = 4,
        num_layers: int = 2,
        attn_dropout: float = 0.1,   # Transformer 内部的 dropout
        ff_multiplier: float = 2.0,
        use_attn_pool: bool = True,
        mlp_dropout: float = 0.1,    # per-point MLP dropout
        feat_dropout: float = 0.1    # self-attn 之后的 dropout
    ):
        super().__init__()
        self.use_attn_pool = use_attn_pool

        # 1) per-point MLP: (B, N, in_dim) -> (B, N, embed_dim)
        mlp_layers = []
        dims = [in_dim] + list(hidden_dims) + [embed_dim]
        for i in range(len(dims) - 1):
            mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:  # 最后一层不加激活和 dropout
                mlp_layers.append(nn.ReLU(inplace=True))
                if mlp_dropout > 0:
                    mlp_layers.append(nn.Dropout(mlp_dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        # 2) Self-Attention：沿着 N 维，让每个点看到所有其他点
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * ff_multiplier),
            dropout=attn_dropout,
            batch_first=True,
            activation="relu",
            norm_first=True
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # self-attn 之后再丢一点特征
        self.feat_dropout = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()

        # 3) Pooling head：attention pooling（可学习） or max pooling（PointNet风格）
        if self.use_attn_pool:
            # 每个点 -> 一个标量 score；softmax 后做加权求和
            self.pool_score = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C_in)
        return:
            global_feat: (B, embed_dim)
            local_feat:  (B, N, embed_dim)
        """
        # per-point MLP
        feat = self.mlp(x)               # (B, N, embed_dim)

        # self-attention
        feat = self.self_attn(feat)      # (B, N, embed_dim)
        feat = self.feat_dropout(feat)   # (B, N, embed_dim)
        local_feat = feat

        # global pooling
        if self.use_attn_pool:
            # scores: (B, N, 1) -> weights: (B, N, 1)
            scores = self.pool_score(local_feat)
            weights = torch.softmax(scores, dim=1)
            # weighted sum over N -> (B, embed_dim)
            global_feat = torch.sum(weights * local_feat, dim=1)
        else:
            # maxpool over N -> (B, embed_dim)
            global_feat = torch.max(local_feat, dim=1)[0]

        return global_feat, local_feat

import math

def knn_indices(feat: torch.Tensor, k: int):
    """
    feat: (B, N, C)
    return idx: (B, N, k)  每个点在 feature 空间的 kNN
    """
    with torch.no_grad():
        # pairwise distance in feature space
        # dist(b,i,j) = ||fi - fj||^2
        B, N, C = feat.shape
        # (B, N, 1, C) - (B, 1, N, C) => (B, N, N)
        dist = torch.cdist(feat, feat, p=2)  # (B, N, N)
        idx = dist.topk(k=k + 1, largest=False)[1]  # (B, N, k+1) include itself
        idx = idx[:, :, 1:]  # remove itself
    return idx

def index_points(points: torch.Tensor, idx: torch.Tensor):
    """
    points: (B, N, C)
    idx: (B, M, k) or (B, N, k)
    return: (B, M, k, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape.append(points.shape[-1])
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1])
    return torch.gather(points.unsqueeze(1).expand(-1, idx.shape[1], -1, -1), 2, idx_expand)

class LocalAttentionBlock(nn.Module):
    """
    只在 kNN 邻域内做 attention (O(Nk))
    """
    def __init__(self, dim, num_heads=4, attn_dropout=0.1, ff_multiplier=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.ln1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * ff_multiplier)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * ff_multiplier), dim)
        )
        self.ff_drop = nn.Dropout(attn_dropout)

    def forward(self, x, neigh_x):
        """
        x:       (B, N, C)  center features
        neigh_x: (B, N, k, C) neighbor features
        """
        B, N, C = x.shape
        k = neigh_x.shape[2]

        # PreNorm
        x_norm = self.ln1(x)
        neigh_norm = self.ln1(neigh_x)

        # Q: (B,N,H,D), K/V: (B,N,k,H,D)
        q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim)
        k_feat = self.k_proj(neigh_norm).view(B, N, k, self.num_heads, self.head_dim)
        v_feat = self.v_proj(neigh_norm).view(B, N, k, self.num_heads, self.head_dim)

        # attn: (B,N,H,k)
        attn = (q.unsqueeze(2) * k_feat).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # weighted sum -> (B,N,H,D)
        out = (attn.unsqueeze(-1) * v_feat).sum(dim=2)
        out = out.reshape(B, N, C)
        out = self.out_proj(out)

        # Residual 1
        x = x + out

        # FFN
        ff_in = self.ln2(x)
        x = x + self.ff_drop(self.ff(ff_in))
        return x

class AttentionPointNet2(nn.Module):
    """
    AttentionPointNet++ (局部注意力版)
    输入:  x (B, N, C_in)  (这里 C_in= joint_feat + env_expand = 128)
    输出:  global_feat (B, embed_dim)
          local_feat  (B, N, embed_dim)
    """
    def __init__(
        self,
        in_dim: int = 128,
        embed_dim: int = 256,
        hidden_dims=[256, 256],
        num_heads: int = 4,
        num_layers: int = 2,
        k: int = 16,
        attn_dropout: float = 0.1,
        ff_multiplier: float = 2.0,
        mlp_dropout: float = 0.0,
        feat_dropout: float = 0.1,
        use_attn_pool: bool = True,
    ):
        super().__init__()
        self.k = k
        self.use_attn_pool = use_attn_pool

        # 1) per-point embedding MLP
        dims = [in_dim] + list(hidden_dims) + [embed_dim]
        mlp_layers = []
        for i in range(len(dims) - 1):
            mlp_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                mlp_layers.append(nn.ReLU(inplace=True))
                if mlp_dropout > 0:
                    mlp_layers.append(nn.Dropout(mlp_dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        # 2) 多层局部 attention（替代全局 TransformerEncoder）
        self.blocks = nn.ModuleList([
            LocalAttentionBlock(embed_dim, num_heads=num_heads,
                                attn_dropout=attn_dropout,
                                ff_multiplier=ff_multiplier)
            for _ in range(num_layers)
        ])

        self.feat_dropout = nn.Dropout(feat_dropout) if feat_dropout > 0 else nn.Identity()

        # 3) pooling head
        if use_attn_pool:
            self.pool_score = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C_in)
        return:
            global_feat: (B, embed_dim)
            local_feat:  (B, N, embed_dim)
        """
        # embed
        feat = self.mlp(x)  # (B,N,embed_dim)

        # local attention message passing
        # kNN is computed in feature space (can be replaced by xyz-based if available)
        idx = knn_indices(feat, k=self.k)          # (B,N,k)
        neigh = index_points(feat, idx)            # (B,N,k,C)

        for blk in self.blocks:
            feat = blk(feat, neigh)
            feat = self.feat_dropout(feat)
            # recompute neighborhood after update (optional but usually helps)
            idx = knn_indices(feat, k=self.k)
            neigh = index_points(feat, idx)

        local_feat = feat

        # global pooling
        if self.use_attn_pool:
            scores = self.pool_score(local_feat)          # (B,N,1)
            weights = torch.softmax(scores, dim=1)
            global_feat = torch.sum(weights * local_feat, dim=1)
        else:
            global_feat = torch.max(local_feat, dim=1)[0]

        return global_feat, local_feat
