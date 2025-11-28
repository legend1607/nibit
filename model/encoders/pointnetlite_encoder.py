import torch
import torch.nn as nn

from model.modules.builders import build_shared_mlp

class PointNetLiteEncoder(nn.Module):
    def __init__(self, in_dim=3, embed_dim=1024, hidden_dims=[64, 128], conv_dim=1, return_all=False):
        """
        conv_dim (int): default=1; set to 2 if working with grouped points (B, G, S, C)
        return_all (bool): If True, return global_feat (B, embed_dim), local_feat (B, N, embed_dim)
        """
        super().__init__()
        self.grouped = False if conv_dim == 1 else True
        self.return_all = return_all
        self.mlp = build_shared_mlp([in_dim] + hidden_dims + [embed_dim], conv_dim=conv_dim)

    def forward(self, x):   # (B, N, in_dim) or grouped points (B, G, S, in_dim)
        if self.grouped:
            x = x.permute(0, 3, 2, 1)     # (B, in_dim, S, G)
        else:
            x = x.permute(0, 2, 1)        # (B, in_dim, N)
            

        feat = self.mlp(x)   # (B, embed_dim, N) or (B, embed_dim, S, G)
        global_feat = torch.max(feat, dim=2)[0]  # maxpool over N/S; (B, embed_dim, (G))

        if self.grouped:
            global_feat = global_feat.permute(0, 2, 1)   # (B, G, embed_dim)

        if self.return_all:
            if self.grouped:
                local_feat = feat.permute(0, 3, 2, 1)  # (B, G, S, embed_dim)
            else:
                local_feat = feat.permute(0, 2, 1)  # (B, N, embed_dim)
            return global_feat, local_feat
            
        return global_feat
