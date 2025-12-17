import torch
import torch.nn as nn

class Encoder_CNN_3D(nn.Module):
    def __init__(self, token_dim=64, token_grid=(2, 2, 1)):  # 2*2*1 = 4 tokens
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 2, 3, 1, 1), nn.BatchNorm3d(2), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(2, 4, 3, 1, 1), nn.BatchNorm3d(4), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(4, 8, 3, 1, 1), nn.BatchNorm3d(8), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, 1, 1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(),
        )

        # 变成 4 个空间块 -> 4 tokens
        self.token_pool = nn.AdaptiveAvgPool3d(token_grid)   # (B,32,2,2,1)
        self.token_proj = nn.Conv3d(32, token_dim, 1)        # (B,64,2,2,1)

    def forward(self, x, return_tokens=False):
        feat_map = self.conv_layers(x)               # (B,32,D,H,W)
        tok_map  = self.token_pool(feat_map)         # (B,32,2,2,1)
        tok_map  = self.token_proj(tok_map)          # (B,64,2,2,1)

        # (B, T=4, 64)
        env_tokens = tok_map.flatten(2).transpose(1, 2)

        # 一个全局向量（可选，但常用来 concat 给 pointnet）
        env_global = env_tokens.mean(dim=1)          # (B,64)

        if return_tokens:
            return env_global, env_tokens
        return env_global
