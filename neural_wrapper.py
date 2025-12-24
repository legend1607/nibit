import numpy as np
import torch

# 你原工程里应当已有这两个符号：
# - voxelize_env(env_range, obstacles, voxel_resolution)
# - JointPointNetEncoder
from model.ECSP.encoders.joint_pointlite_encoder import JointPointNetEncoder
from demo_planning_arm import voxelize_env  # 如果你的 voxelize_env 在别处，请改这里


class ECSP_NeuralWrapper:
    """
    Inference wrapper for JointPointNetEncoder used by the planner.

    语义（与 train.py / single_test.py 保持一致）：
      - 模型 forward_with_env_feat 返回: (free_logit, path_logit, local_feat)，其中 free_logit/path_logit 形状为 (B, N)
      - P(free)  = sigmoid(free_logit)   （labels: 1=free, 0=collision）
      - P(path)  = sigmoid(path_logit)
    """

    def __init__(self, problem, ckpt_path, voxel_resolution=(50, 50, 50), device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        env_record = problem["env_dict"]

        # dof
        start = np.array(problem["start"], dtype=np.float32)
        self.dof = int(start.shape[0])

        # pose range for normalization
        pose_range = np.array(env_record["pose_range"], dtype=np.float32)  # (dof, 2)
        self.low = pose_range[:, 0]
        self.high = pose_range[:, 1]
        self.span = self.high - self.low
        self.span[self.span == 0] = 1e-6

        # voxelize env
        env_range = np.array(env_record["env_range"], dtype=np.float32)
        obstacles = env_record["obstacles"]
        voxel = voxelize_env(env_range, obstacles, np.array(voxel_resolution, dtype=int)).astype(np.float32)
        self.env_voxel = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,D,H,W)

        # model
        self.model = JointPointNetEncoder(joint_in_dim=4).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # cache env feat
        with torch.no_grad():
            self.env_feat = self.model.encode_env(self.env_voxel)

    @torch.no_grad()
    def _normalize_joints(self, joints_np: np.ndarray) -> np.ndarray:
        q = np.asarray(joints_np, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        q_norm = (q - self.low) / self.span
        q_norm = np.clip(q_norm, 0.0, 1.0)
        return q_norm

    @torch.no_grad()
    def _forward_batch(self, joints_np: np.ndarray):
        q_norm = self._normalize_joints(joints_np)           # (N, dof)
        joints_t = torch.from_numpy(q_norm).float().unsqueeze(0).to(self.device)  # (1, N, dof)
        free_logits, pathlogits, _ = self.model.forward_with_env_feat(self.env_feat, joints_t)
        return free_logits[0], pathlogits[0]  # (N,), (N,)

    # ========= 关键新增：一次 forward 同时拿到概率 =========
    @torch.no_grad()
    def predict_probs(self, joints_np: np.ndarray):
        free_logits, pathlogits = self._forward_batch(joints_np)
        p_free = torch.sigmoid(free_logits).detach().cpu().numpy().astype(np.float32).reshape(-1)
        p_path = torch.sigmoid(pathlogits).detach().cpu().numpy().astype(np.float32).reshape(-1)
        return p_free, p_path
    
    @torch.no_grad()
    def get_free_mask(self, joints_np, prob_th: float = 0.5):
        p_free, _ = self.predict_probs(joints_np)
        free_mask = (p_free >= prob_th)
        return free_mask.astype(bool), p_free

    @torch.no_grad()
    def get_path_mask(self, joints_np, prob_th: float = 0.5):
        _, p_path = self.predict_probs(joints_np)
        path_mask = (p_path >= prob_th)
        return path_mask.astype(bool), p_path

    @torch.no_grad()
    def get_safe_path_mask(self, joints_np, free_th: float = 0.5, path_th: float = 0.5):
        p_free, p_path = self.predict_probs(joints_np)
        safe = (p_free >= free_th) & (p_path >= path_th)
        return safe.astype(bool), p_path
