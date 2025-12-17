"""
single_test.py

可视化脚本：对比模型预测与真实标签（带 Collision 约束），并统计 path label / 预测概率的区间分布。

适配新版 JointPointNetEncoder：
- forward 返回 (coll_logit, path_logit, local_feat)
  - coll_logit: (B, N)    → sigmoid 后为 “碰撞概率”（1=collision）
  - path_logit: (B, N)    → sigmoid 后为 “路径接近程度” 概率 [0,1]
- 训练时：
  - dataset.labels: 0 = collision, 1 = free
  - coll_label = (labels == 0).float()  → 1 表示碰撞
  - path_soft 为 [0,1] 的软标签，碰撞点已被清零

这里测试脚本内部仍然用 0=collision, 1=free 的编码，
只是从 coll_logit 还原预测标签时作一次映射。

"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import argparse
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

from dataset import ArmPointCloudDataset
from model.encoders.joint_pointlite_encoder import JointPointNetEncoder


# ===========================
# 工具函数：区间统计
# ===========================
def print_interval_stats(values, bins, title="Interval Stats"):
    """
    对一维数组 values 按给定 bins 做区间统计并打印。

    参数
    ----
    values : np.ndarray or list
        要统计的一维数据，例如 GT 的 pathlabels 或 Pred 的 path 概率
    bins : list[float]
        区间边界，例如 [0, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
        将会统计 [bins[i], bins[i+1]) 每个区间的数量和占比
    title : str
        标题，用于打印分隔信息
    """
    values = np.asarray(values)
    N = len(values)
    if N == 0:
        print(f"\n=== {title} ===")
        print("空数组，无法统计")
        print("=" * 50)
        return

    hist, _ = np.histogram(values, bins=bins)

    print(f"\n=== {title} ===")
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        count = hist[i]
        ratio = count / N
        print(f"[{left: .6f}, {right: .6f}): {count:6d} ({ratio:.4f})")
    print("=" * 50)


# ===========================
# 模型加载（适配新 train.py 的 checkpoint）
# ===========================
def load_model(ckpt_path, joint_in_dim, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = JointPointNetEncoder(joint_in_dim=joint_in_dim).to(device)

    # 🔑 IMPORTANT FIX
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ===========================
# 标签分类（带 Collision 约束）
# ===========================
def categorize_points_with_constraint(labels_01, pathvalues, path_threshold):
    """
    根据 collision/free 标签 (0=collision,1=free) 和 path 数值进行分类。

    ⭐ 关键约束：
        - Collision 点一定不是 Path
        - 只有 Free 点才可能被视为 Path

    参数
    ----
    labels_01 : np.ndarray / torch.Tensor, shape (N,)
        0 = collision, 1 = free
    pathvalues : np.ndarray / torch.Tensor, shape (N,)
        可以是 GT 的 path soft label，也可以是预测的 path 概率
    path_threshold : float
        阈值，大于等于该值的点会被视为 “path 候选”，再与 free 约束结合。

    返回
    ----
    collision_mask, free_mask, path_mask  (均为 bool mask)
    """
    if torch.is_tensor(labels_01):
        labels_01 = labels_01.cpu().numpy()
    if torch.is_tensor(pathvalues):
        pathvalues = pathvalues.cpu().numpy()

    labels_01 = np.asarray(labels_01)
    pathvalues = np.asarray(pathvalues)

    collision_mask = (labels_01 == 0)
    free_mask = (labels_01 == 1)

    path_mask = (pathvalues >= path_threshold) & free_mask

    return collision_mask, free_mask, path_mask


# ===========================
# 可视化：GT vs Pred
# ===========================
def visualize_comparison(joint_states,
                         gt_labels_01, gt_path_soft,
                         pred_labels_01, pred_path_prob,
                         path_threshold, sample_idx, save_path=None):
    """
    并排可视化真值和预测值（带 Collision 约束）.

    参数中的 labels 编码统一为：0=collision,1=free
    """
    N = joint_states.shape[0]

    # 转 numpy
    joint_states = joint_states.cpu().numpy()
    gt_labels_01 = gt_labels_01.cpu().numpy()
    gt_path_soft = gt_path_soft.cpu().numpy()
    pred_labels_01 = pred_labels_01.cpu().numpy()
    pred_path_prob = pred_path_prob.cpu().numpy()

    # 分类点（真值）
    gt_coll_mask, gt_free_mask, gt_path_mask = categorize_points_with_constraint(
        gt_labels_01, gt_path_soft, path_threshold
    )

    # 分类点（预测）
    pred_coll_mask, pred_free_mask, pred_path_mask = categorize_points_with_constraint(
        pred_labels_01, pred_path_prob, path_threshold
    )

    # 统计“本来 path>=thr 但被 collision 排除”的点（仅作信息显示）
    gt_excluded = ((gt_path_soft >= path_threshold) & (gt_labels_01 == 0)).sum()
    pred_excluded = ((pred_path_prob >= path_threshold) & (pred_labels_01 == 0)).sum()

    # 创建图形
    fig = plt.figure(figsize=(14, 6))

    # === 左图：Ground Truth ===
    ax1 = fig.add_subplot(121, projection='3d')

    # 使用前3个关节角度作为3D坐标
    if joint_states.shape[1] >= 3:
        x, y, z = joint_states[:, 0], joint_states[:, 1], joint_states[:, 2]
    else:
        x = joint_states[:, 0] if joint_states.shape[1] > 0 else np.zeros(N)
        y = joint_states[:, 1] if joint_states.shape[1] > 1 else np.zeros(N)
        z = np.zeros(N)

    gt_free_not_path = gt_free_mask & ~gt_path_mask
    if gt_free_not_path.sum() > 0:
        ax1.scatter(
            x[gt_free_not_path], y[gt_free_not_path], z[gt_free_not_path],
            c='lightgreen', s=20, alpha=0.3, label='Free'
        )

    if gt_coll_mask.sum() > 0:
        ax1.scatter(
            x[gt_coll_mask], y[gt_coll_mask], z[gt_coll_mask],
            c='red', s=20, alpha=0.5, label='Collision'
        )

    if gt_path_mask.sum() > 0:
        ax1.scatter(
            x[gt_path_mask], y[gt_path_mask], z[gt_path_mask],
            c='blue', s=30, alpha=0.8,
            label=f'Path (Free & ≥{path_threshold:.2f})'
        )

    ax1.set_xlabel('Joint 1')
    ax1.set_ylabel('Joint 2')
    ax1.set_zlabel('Joint 3')
    title1 = f'Ground Truth (Sample {sample_idx})\n'
    title1 += f'Free: {gt_free_not_path.sum()}, Coll: {gt_coll_mask.sum()}, Path: {gt_path_mask.sum()}'
    if gt_excluded > 0:
        title1 += f'\n(Excluded {gt_excluded} Coll+Path points)'
    ax1.set_title(title1, fontsize=12)
    ax1.legend()

    # === 右图：Prediction ===
    ax2 = fig.add_subplot(122, projection='3d')

    pred_free_not_path = pred_free_mask & ~pred_path_mask
    if pred_free_not_path.sum() > 0:
        ax2.scatter(
            x[pred_free_not_path], y[pred_free_not_path], z[pred_free_not_path],
            c='lightgreen', s=20, alpha=0.3, label='Free'
        )

    if pred_coll_mask.sum() > 0:
        ax2.scatter(
            x[pred_coll_mask], y[pred_coll_mask], z[pred_coll_mask],
            c='red', s=20, alpha=0.5, label='Collision'
        )

    if pred_path_mask.sum() > 0:
        ax2.scatter(
            x[pred_path_mask], y[pred_path_mask], z[pred_path_mask],
            c='blue', s=30, alpha=0.8,
            label=f'Path (Free & ≥{path_threshold:.2f})'
        )

    ax2.set_xlabel('Joint 1')
    ax2.set_ylabel('Joint 2')
    ax2.set_zlabel('Joint 3')
    title2 = f'Prediction (Sample {sample_idx})\n'
    title2 += f'Free: {pred_free_not_path.sum()}, Coll: {pred_coll_mask.sum()}, Path: {pred_path_mask.sum()}'
    if pred_excluded > 0:
        title2 += f'\n(Excluded {pred_excluded} Coll+Path points)'
    ax2.set_title(title2, fontsize=12)
    ax2.legend()

    # 统计信息文本
    stats_text = f"""
Statistics (With Collision Constraint):
GT - Collision: {gt_coll_mask.sum()}
GT - Free: {gt_free_mask.sum()}
GT - Path (Free only): {gt_path_mask.sum()}

Pred - Collision: {pred_coll_mask.sum()}
Pred - Free: {pred_free_mask.sum()}
Pred - Path (Free only): {pred_path_mask.sum()}
"""
    plt.figtext(
        0.5, 0.02, stats_text, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


# ===========================
# 可视化：GT / Pred Path 分布 + 区间统计
# ===========================
def visualize_pathlabel_distribution(pathlabels, pred_pathlabels,
                                     sample_idx, save_path=None,
                                     show_interval_stats=True):
    """
    可视化 GT path label 与模型预测 path 概率的分布，
    并（默认）打印区间统计。
    """
    pathlabels = np.asarray(pathlabels)
    pred_pathlabels = np.asarray(pred_pathlabels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Ground Truth 分布 ---
    ax1 = axes[0]
    ax1.hist(pathlabels, bins=80, alpha=0.7, edgecolor='black', label='GT Soft Label')
    ax1.set_title(f'GT Path Label Distribution (Sample {sample_idx})')
    ax1.set_xlabel('GT Path Label Value')
    ax1.set_ylabel('Count')
    ax1.set_xlim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()

    # --- Prediction 分布 ---
    ax2 = axes[1]
    ax2.hist(pred_pathlabels, bins=80, alpha=0.7, edgecolor='black', label='Predicted Probability')
    ax2.set_title(f'Pred Path Probability Distribution (Sample {sample_idx})')
    ax2.set_xlabel('Predicted Path Probability')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        dist_path = save_path.replace('.png', '_distribution.png')
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        print(f"Distribution saved to: {dist_path}")

    plt.show()

    # ====== 区间统计（默认开启） ======
    if show_interval_stats:
        bins = [0.0, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.6,0.7,0.8,0.9, 1.0 + 1e-6]

        print_interval_stats(
            pathlabels,
            bins,
            title=f"GT Path Soft Label 区间统计 (Sample {sample_idx})"
        )
        print_interval_stats(
            pred_pathlabels,
            bins,
            title=f"Pred Path Probability 区间统计 (Sample {sample_idx})"
        )


# ===========================
# 主函数
# ===========================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions (with Collision constraint)"
    )

    # 模型和数据
    parser.add_argument(
        '--checkpoint', type=str,
        default='results/model_training/train_20251215-144528/best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_path', type=str,
        default='data/liche/val/val.npz',
        help='Path to dataset'
    )

    # 可视化参数
    parser.add_argument(
        '--sample_idx', type=int, default=110,
        help='Index of sample to visualize'
    )
    parser.add_argument(
        '--path_threshold', type=float, default=0.8,
        help='Path label / probability threshold，用于定义 path 区域'
    )
    parser.add_argument(
        '--coll_threshold', type=float, default=0.7,
        help='Collision 概率阈值，p_coll > 该值视为碰撞'
    )

    # 输出
    parser.add_argument(
        '--save_dir', type=str,
        default='results/visualizations',
        help='Directory to save visualizations'
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading dataset from: {args.data_path}")
    dataset = ArmPointCloudDataset(
        npz_path=args.data_path,
        max_points=None,
        shuffle_points=False,
        device=None,
        require_pathlabels=True,
    )
    print(f"Dataset size: {len(dataset)}")

    if args.sample_idx >= len(dataset):
        print(f"Error: sample_idx {args.sample_idx} >= dataset size {len(dataset)}")
        return

    joint_in_dim = dataset.dof
    model = load_model(args.checkpoint, joint_in_dim, device)

    path_threshold = args.path_threshold
    print(f"\nUsing path threshold: {path_threshold:.3f}")
    print(f"Using collision threshold: {args.coll_threshold:.3f}")

    # 获取样本
    print(f"\nProcessing sample {args.sample_idx}...")
    env_voxel, joint_states, labels, pathlabels, meta = dataset[args.sample_idx]

    # 添加 batch 维度
    env_voxel = env_voxel.unsqueeze(0).to(device)     # (1, 1, D, H, W)
    joint_states = joint_states.unsqueeze(0).to(device)  # (1, N, DoF)
    labels = labels.to(device)                        # (N,) 0=collision,1=free
    pathlabels = pathlabels.to(device)                # (N,) soft [0,1]

    # 模型推理
    with torch.no_grad():
        coll_logit, path_logit, _ = model(env_voxel, joint_states)

    coll_logit = coll_logit.squeeze(0)    # (N,)
    path_logit = path_logit.squeeze(0)    # (N,)

    # 概率
    p_coll = torch.sigmoid(coll_logit)       # 碰撞概率（1=collision）
    p_path = torch.sigmoid(path_logit)       # 路径概率
    # ===== DEBUG: only GT>=0.7 points =====
    ps = pathlabels.view(-1)          # 注意：这里用 GT path_soft
    pp = p_path.view(-1)

    high = ps >= 0.7
    print(
        "GT>=0.7 cnt:", int(high.sum()),
        "max pred on GT>=0.7:",
        float(pp[high].max()) if high.any() else None,
        "mean pred on GT>=0.7:",
        float(pp[high].mean()) if high.any() else None,
        "max pred overall:",
        float(pp.max())
    )

    # ======== 预测标签（0=collision, 1=free）========
    # p_coll > coll_threshold → 认为是 collision → label=0
    pred_is_free = (p_coll > args.coll_threshold)
    pred_labels_01 = torch.where(
        pred_is_free,
        torch.ones_like(p_coll, dtype=torch.long),
        torch.zeros_like(p_coll, dtype=torch.long)
    )

    # === 统计（带约束） ===
    N = labels.shape[0]

    # Ground Truth（0=collision,1=free）
    gt_coll_mask, gt_free_mask, gt_path_mask_constrained = categorize_points_with_constraint(
        labels, pathlabels, path_threshold
    )

    # Prediction
    pred_coll_mask, pred_free_mask, pred_path_mask_constrained = categorize_points_with_constraint(
        pred_labels_01, p_path, path_threshold
    )

    print(f"\n{'='*70}")
    print(f"Sample {args.sample_idx} statistics (带 Collision 约束):")

    print(f"\n{'Ground Truth:':-^70}")
    print(f"  Collision: {gt_coll_mask.sum()} "
          f"({gt_coll_mask.sum()/N*100:.1f}%)")
    print(f"  Free: {gt_free_mask.sum()}")
    print(f"  Path (Free & ≥thr): {gt_path_mask_constrained.sum()}")

    print(f"\n{'Prediction:':-^70}")
    print(f"  Collision: {pred_coll_mask.sum()} "
          f"({pred_coll_mask.sum()/N*100:.1f}%)")
    print(f"  Free: {pred_free_mask.sum()}")
    print(f"  Path (Free & ≥thr): {pred_path_mask_constrained.sum()}")

    # Path RMSE（不加阈值，直接 soft vs prob）
    path_rmse = torch.sqrt(torch.mean((p_path - pathlabels) ** 2)).item()
    print(f"\n  Path RMSE (all points): {path_rmse:.6f}")

    # 带约束的 Path 性能（简单 F1）
    if gt_path_mask_constrained.sum() > 0:
        path_constrained_tp = (pred_path_mask_constrained & gt_path_mask_constrained).sum().item()
        precision = path_constrained_tp / (pred_path_mask_constrained.sum().item() + 1e-8)
        recall = path_constrained_tp / (gt_path_mask_constrained.sum().item() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"  Path F1 (Constrained): {f1:.4f}")
        print(f"  Path Precision (Constrained): {precision:.4f}")
        print(f"  Path Recall (Constrained):    {recall:.4f}")

    print(f"{'='*70}\n")

    # 可视化 GT vs Pred
    save_path = os.path.join(args.save_dir, f'sample_{args.sample_idx}_constrained.png')

    visualize_comparison(
        joint_states.squeeze(0),   # (N, DoF)
        labels,                    # GT 0/1
        pathlabels,
        pred_labels_01,
        p_path,
        path_threshold,
        args.sample_idx,
        save_path
    )

    # 默认：可视化 GT/Pred Path 分布 + 区间统计（终端打印）
    visualize_pathlabel_distribution(
        pathlabels.cpu().numpy(),
        p_path.cpu().numpy(),
        args.sample_idx,
        save_path
    )

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
