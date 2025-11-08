from os.path import join
import torch
import numpy as np
from model.pointnet2ce import PointNet2DualHeadAttention
from model.pointnet2_utils import pc_normalize

class PNGWrapper:
    def __init__(
        self,
        root_dir='.',
        coord_dim=2,
        device='cuda',
    ):
        """
        初始化双头 PointNet++ 模型
        """
        self.device = device
        self.model = PointNet2DualHeadAttention(coord_dim=coord_dim, feature_dim=3).to(device)  # dual-head
        model_filepath = join(
            root_dir,
            'results/model_training/random_pointnet2ce_2d/checkpoints/best_random_pointnet2ce_2d.pth'
        )
        checkpoint = torch.load(model_filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("PointNet++ dual-head wrapper initialized.")

    def predict_probabilities(self, pc, start_mask, goal_mask):
        """
        - inputs:
            - pc: np.float32 (n_points, 2 or 3)
            - start_mask: np.float32 (n_points,)
            - goal_mask: np.float32 (n_points,)
        - outputs:
            - path_prob: np.float32 (n_points,), 属于路径的概率
            - keypoint_prob: np.float32 (n_points,), 属于关键点的概率
        """
        with torch.no_grad():
            n_points = pc.shape[0]


            # 归一化点云
            pc_xyz = torch.from_numpy(pc_normalize(pc)).float().to(self.device)

            # 特征拼接
            free_mask = 1 - (start_mask + goal_mask).astype(bool)
            pc_features = np.stack(
                (start_mask, goal_mask, free_mask.astype(np.float32)),
                axis=-1
            )
            pc_features = torch.from_numpy(pc_features).float().to(self.device)

            # 输入模型
            model_input = torch.cat([pc_xyz, pc_features], dim=1).unsqueeze(0).permute(0, 2, 1)  # (1, 6, N)
            
            # 前向
            path_logits, keypoint_logits = self.model(model_input)  # (1,2,N) each
            path_prob = torch.sigmoid(path_logits)
            keypoint_prob = torch.sigmoid(keypoint_logits)


            return path_prob, keypoint_prob
