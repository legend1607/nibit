import torch
import torch.nn as nn
from model.modules.builders import build_fc_layers


class JointMLPEncoder(nn.Module):
    """
    N 个关节点 → MLP → joint_feat (B, N, F_joint)
    joint_states: (B, N, joint_in_dim)
    输出 joint_feat: (B, N, joint_feat_dim)
    """
    def __init__(
        self,
        joint_in_dim=7,         # 关节维度（比如 7 自由度）
        joint_feat_dim=64,      # 输出 F_joint
        hidden_dims=[128, 128], # 中间隐层
        dropout=0.1
    ):
        super().__init__()
        # 用你自己的 build_fc_layers 搭 MLP: in → hidden → ... → F_joint
        self.mlp = build_fc_layers(
            [joint_in_dim] + hidden_dims + [joint_feat_dim],
            dropout=dropout
        )

    def forward(self, joint_states):
        """
        joint_states: (B, N, joint_in_dim)
        """
        B, N, C = joint_states.shape
        x = joint_states.reshape(B * N, C)  # 展平成 (B*N, joint_in_dim)
        x = self.mlp(x)                     # (B*N, joint_feat_dim)
        joint_feat = x.reshape(B, N, -1)    # 再 reshape 回 (B, N, F_joint)
        return joint_feat
