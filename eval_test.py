import os
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from dataset import ArmPointCloudDataset, collate_fn_with_meta
from model.encoders.joint_pointlite_encoder import JointPointNetEncoder

import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import time  # 用于计时


# ================================================================
# argparse
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser("ArmPlanningEval")

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='results/model_training/liche_pointnet/checkpoints/best_model.pt'
    )
    parser.add_argument('--num_vis', type=int, default=6, help='number of samples to visualize')
    parser.add_argument(
        '--max_points_vis',
        type=int,
        default=1000,
        help='max points per sample to scatter in visualization'
    )

    # eval 随机环境 & 随机关节
    parser.add_argument(
        '--eval_num_env',
        type=int,
        default=None,
        help='if set, randomly choose this many envs from test set for evaluation'
    )
    parser.add_argument(
        '--max_points_eval',
        type=int,
        default=None,
        help='if set, randomly choose this many joint states per env for evaluation'
    )

    # ★ 明确区分 单任务 / 多任务 / 两者都跑
    parser.add_argument(
        '--eval_mode',
        type=str,
        default='single',
        choices=['single', 'multi', 'both'],
        help='single: 只跑单任务（单环境延迟 benchmark）；'
             'multi: 只跑多任务（全 test 集评估）；'
             'both: 两者都跑'
    )

    return parser.parse_args()


# ================================================================
# 单样本 benchmark：随机环境 + 随机关节
# 每个 voxel 只算一次 env_feat，后面只跑 forward_with_env_feat
# ================================================================
def benchmark_single_sample(
    model,
    dataset,
    device,
    log_fn=print,
    warmup=10,
    num_run=50,
    num_env=1,
    max_points=None,
    near_thresh=0.5,
    vis_out_dir=None,
):
    """
    单任务（Single-task）延迟 benchmark，label 为 collision/free (0/1)，并可视化 path_gt/path_pred。

    - 随机选 num_env 个 env；
    - 每个 env 随机选最多 max_points 个关节角度点；
    - 对每个 env，只算一次 env_feat，多次前向 head 计时；
    - path_gt  : 由 GT distance (dist_gt < near_thresh) 决定；
    - path_pred: 由模型预测 distance (dist_pred < near_thresh) 决定。
    - 可视化：左图 path_gt，右图 path_pred。
    """
    model.eval()
    import random
    indices = random.sample(range(len(dataset)), k=min(num_env, len(dataset)))

    cnn_times = []
    head_times = []
    M_record = None

    if vis_out_dir is None:
        vis_out_dir = os.path.join(
            "results", "model_training", "liche_pointnet", "figures", "single_task_ispath"
        )
    os.makedirs(vis_out_dir, exist_ok=True)

    for sample_idx, idx in enumerate(indices):
        sample = dataset[idx]

        # 假设 sample = (env_voxel, joint_states, labels, distances, meta)
        if len(sample) == 5:
            env_voxel, joint_states, labels, distances, meta = sample
        elif len(sample) == 4:
            env_voxel, joint_states, labels, meta = sample
            distances = None
        else:
            raise ValueError(f"Unexpected sample length {len(sample)} from dataset[idx].")

        N = joint_states.shape[0]

        # 子采样点
        if max_points is not None and N > max_points:
            perm = torch.randperm(N)[:max_points]
            joint_states = joint_states[perm]
            labels = labels[perm]
            if distances is not None:
                distances = distances[perm]
            N = joint_states.shape[0]

        env_voxel_b = env_voxel.unsqueeze(0).to(device)        # (1,1,D,H,W)
        joint_states_b = joint_states.unsqueeze(0).to(device)  # (1,M,dof)
        labels_b = labels.to(device)                           # (M,)
        B, M, _ = joint_states_b.shape
        M_record = M

        # GT distance
        if distances is not None:
            dist_gt_b = distances.to(device).view(1, -1)       # (1,M)
        else:
            dist_gt_b = None

        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                env_feat = model.encode_env(env_voxel_b)
                _ = model.forward_with_env_feat(env_feat, joint_states_b)

        # CNN 一次
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            env_feat = model.encode_env(env_voxel_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        cnn_dt = t1 - t0

        # 先跑一次 head 拿 dist_pred
        with torch.no_grad():
            out = model.forward_with_env_feat(env_feat, joint_states_b)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                logits = out[0]
                dist_pred_b = out[1]  # (1,M) or (1,M,1)
            else:
                logits = out
                dist_pred_b = None

            if dist_pred_b is not None:
                dist_pred_b = dist_pred_b.view(1, -1)          # (1,M)

            # 正式计时：只计 head
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            for _ in range(num_run):
                _ = model.forward_with_env_feat(env_feat, joint_states_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.time()
        head_dt = (t3 - t2) / num_run

        cnn_times.append(cnn_dt)
        head_times.append(head_dt)

        log_fn(
            f"[SINGLE-TASK] Env idx={idx}: "
            f"M={M}, cnn={cnn_dt*1000:.3f}ms, head={head_dt*1000:.3f}ms"
        )

        # === 可视化：左 path_gt，右 path_pred ===
        dist_gt_1d = dist_gt_b.squeeze(0) if dist_gt_b is not None else None
        dist_pred_1d = dist_pred_b.squeeze(0) if dist_pred_b is not None else None

        visualize_ispath_two_panel(
            joint_states=joint_states,  # (M,dof), CPU tensor
            labels=labels,              # (M,),   CPU tensor
            dist_gt=dist_gt_1d.cpu() if dist_gt_1d is not None else None,
            dist_pred=dist_pred_1d.cpu() if dist_pred_1d is not None else None,
            out_dir=vis_out_dir,
            sample_idx=sample_idx,
            max_points_vis=max_points if max_points is not None else 1000,
            env_idx=idx,
            near_thresh=near_thresh,
        )

    # 统计时间
    avg_cnn = sum(cnn_times) / len(cnn_times)
    avg_head = sum(head_times) / len(head_times)

    per_point_cnn = avg_cnn / (B * M_record)
    per_point_head = avg_head / (B * M_record)

    log_fn("========== [SINGLE-TASK] 单环境多关节延迟 (ispath by distance) ==========")
    log_fn(f"Env indices             : {indices}")
    log_fn(f"Num points per env      : {M_record} (after random sampling)")
    log_fn(f"Avg CNN time            : {avg_cnn * 1000:.3f} ms / env")
    log_fn(f"Avg head time           : {avg_head * 1000:.3f} ms / env")
    log_fn(f"Avg total (CNN+head)    : {(avg_cnn + avg_head) * 1000:.3f} ms / env")
    log_fn(f"Avg CNN per-point       : {per_point_cnn * 1e6:.3f} us / point")
    log_fn(f"Avg head per-point      : {per_point_head * 1e6:.3f} us / point")
    log_fn(f"[SINGLE-TASK] Visualizations saved to: {vis_out_dir}")
    log_fn("=====================================================================")

# ================================================================
# metrics
# ================================================================
def compute_metrics_from_confmat(confmat):
    """
    confmat: (C, C)
    confmat[i, j] = 真实类别 i 被预测为 j 的数量
    """
    num_classes = confmat.shape[0]
    confmat = confmat.float()

    TP = confmat.diag()
    FP = confmat.sum(dim=0) - TP
    FN = confmat.sum(dim=1) - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "per_class": {
            "precision": precision,  # (C,)
            "recall": recall,
            "iou": iou,
            "f1": f1,
        },
        "macro": {
            "precision": precision.mean(),
            "recall": recall.mean(),
            "iou": iou.mean(),
            "f1": f1.mean(),
        }
    }
    return metrics


# ================================================================
# visualization
# ================================================================
def visualize_sample_2d_joint(
    joint_states, labels, preds,
    out_dir, sample_idx,
    max_points_vis=1000,
    env_idx=None,
):
    """
    joint_states: (N, dof) tensor, on cpu
    labels: (N,) tensor, on cpu, int64
    preds: (N,) tensor, on cpu, int64
    env_idx: 该样本在 dataset 中的 index（通过 DataLoader.sampler 推出来）
    """
    os.makedirs(out_dir, exist_ok=True)

    joint_states = joint_states.detach().cpu().numpy()  # (N, dof)
    labels = labels.detach().cpu().numpy().astype(np.int64)
    preds = preds.detach().cpu().numpy().astype(np.int64)

    # 防止有 ignore_index（例如 -1）
    valid = labels >= 0
    joint_states = joint_states[valid]
    labels = labels[valid]
    preds = preds[valid]

    N, dof = joint_states.shape
    if dof < 2:
        print(f"[Warn] dof={dof} < 2, cannot project to 2D (joint_i, joint_j). Skip vis for sample {sample_idx}.")
        return

    # 随机选两个关节维度用于可视化
    dim_x, dim_y = np.random.choice(dof, size=2, replace=False)

    # 选 max_points_vis 个点做可视化，避免太密
    if N > max_points_vis:
        idx = np.random.choice(N, size=max_points_vis, replace=False)
    else:
        idx = np.arange(N)

    x = joint_states[idx, dim_x]
    y = joint_states[idx, dim_y]
    labels_v = labels[idx]
    preds_v = preds[idx]

    # 颜色映射：0=红，1=绿，2=蓝
    color_map = np.array(['red', 'green', 'blue'])
    colors_gt = color_map[labels_v]
    colors_pred = color_map[preds_v]

    # title 里带上 env_idx（如果有的话）
    if env_idx is not None:
        title_prefix = f"[Env {env_idx}] "
    else:
        title_prefix = ""

    # 创建图像
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    # 左：Ground Truth
    axes[0].scatter(x, y, c=colors_gt, s=5, alpha=0.7)
    axes[0].set_title(
        f"{title_prefix}Sample {sample_idx} - Ground Truth\n(joint {dim_x} vs joint {dim_y})"
    )
    axes[0].set_xlabel(f"joint {dim_x}")
    axes[0].set_ylabel(f"joint {dim_y}")

    # 右：Prediction
    axes[1].scatter(x, y, c=colors_pred, s=5, alpha=0.7)
    axes[1].set_title(
        f"{title_prefix}Sample {sample_idx} - Prediction\n(joint {dim_x} vs joint {dim_y})"
    )
    axes[1].set_xlabel(f"joint {dim_x}")
    axes[1].set_ylabel(f"joint {dim_y}")

    # 简单的 legend，用 dummy points
    import matplotlib.lines as mlines
    legend_elems = [
        mlines.Line2D([], [], color='red', marker='o', linestyle='None', label='collision (0)', markersize=5),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', label='free (1)', markersize=5),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='path (2)', markersize=5),
    ]
    axes[0].legend(handles=legend_elems, loc='best', fontsize=8)

    plt.tight_layout()

    # 文件名也带上 env_idx，方便回溯
    if env_idx is not None:
        out_name = f"sample_{sample_idx:04d}_env_{env_idx:04d}.png"
    else:
        out_name = f"sample_{sample_idx:04d}.png"

    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path)
    plt.close(fig)

def visualize_ispath_two_panel(
    joint_states,
    labels,
    dist_gt,        # (N,) or None, GT distance
    dist_pred,      # (N,) or None, predicted distance
    out_dir,
    sample_idx,
    max_points_vis=1000,
    env_idx=None,
    near_thresh=0.5,
):
    """
    适用于 label 只有 collision/free (0/1) 的情况。

    左图: 使用 GT distance 判定 path_gt (dist_gt < near_thresh)
      - collision (label=0)          : red
      - free & 非 path_gt            : green
      - free & path_gt (ispath)      : blue

    右图: 使用模型预测的 dist_pred 判定 path_pred (dist_pred < near_thresh)
      - collision (label=0)          : red
      - free & 非 path_pred          : green
      - free & path_pred (ispath)    : blue
    """
    os.makedirs(out_dir, exist_ok=True)

    # 转 numpy
    joint_states = joint_states.detach().cpu().numpy()  # (N, dof)
    labels = labels.detach().cpu().numpy().astype(np.int64)

    if dist_gt is not None:
        dist_gt = dist_gt.detach().cpu().numpy()
    if dist_pred is not None:
        dist_pred = dist_pred.detach().cpu().numpy()

    # 去掉 ignore_index
    valid = labels >= 0
    joint_states = joint_states[valid]
    labels = labels[valid]
    if dist_gt is not None:
        dist_gt = dist_gt[valid]
    if dist_pred is not None:
        dist_pred = dist_pred[valid]

    N, dof = joint_states.shape
    if dof < 2 or N == 0:
        print(f"[Warn] sample {sample_idx}: dof={dof}, N={N}, skip vis.")
        return

    # 随机选两个关节维度
    dim_x, dim_y = np.random.choice(dof, size=2, replace=False)

    # 子采样
    if N > max_points_vis:
        idx = np.random.choice(N, size=max_points_vis, replace=False)
    else:
        idx = np.arange(N)

    js = joint_states[idx]
    lbl = labels[idx]
    dgt = dist_gt[idx] if dist_gt is not None else None
    dpred = dist_pred[idx] if dist_pred is not None else None

    x = js[:, dim_x]
    y = js[:, dim_y]

    is_collision = (lbl == 0)
    is_free = (lbl == 1)

    if env_idx is not None:
        title_prefix = f"[Env {env_idx}] "
    else:
        title_prefix = ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # ---------------- 左图：path_gt ----------------
    if dgt is not None:
        path_gt = (dgt < near_thresh) & is_free   # 只在 free 点上定义 path_gt
    else:
        path_gt = np.zeros_like(is_free, dtype=bool)

    free_non_path_gt = is_free & (~path_gt)

    colors_left = np.full_like(lbl, fill_value='lightgray', dtype=object)
    colors_left[is_collision] = 'red'
    colors_left[free_non_path_gt] = 'green'
    colors_left[path_gt] = 'blue'   # path_gt 高亮

    axes[0].scatter(x, y, c=colors_left, s=5, alpha=0.7)
    axes[0].set_xlabel(f"joint {dim_x}")
    axes[0].set_ylabel(f"joint {dim_y}")
    axes[0].set_title(
        f"{title_prefix}Sample {sample_idx} - path_gt\n"
        f"(dist_gt < {near_thresh})"
    )

    import matplotlib.lines as mlines
    legend_left = [
        mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                      label='collision (0)', markersize=5),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                      label='free non-path_gt', markersize=5),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                      label='path_gt (free & dist_gt<th)', markersize=5),
    ]
    axes[0].legend(handles=legend_left, loc='best', fontsize=8)

    # ---------------- 右图：path_pred ----------------
    if dpred is not None:
        path_pred = (dpred < near_thresh) & is_free
    else:
        path_pred = np.zeros_like(is_free, dtype=bool)

    free_non_path_pred = is_free & (~path_pred)

    colors_right = np.full_like(lbl, fill_value='lightgray', dtype=object)
    colors_right[is_collision] = 'red'
    colors_right[free_non_path_pred] = 'green'
    colors_right[path_pred] = 'blue'  # path_pred 高亮

    axes[1].scatter(x, y, c=colors_right, s=5, alpha=0.7)
    axes[1].set_xlabel(f"joint {dim_x}")
    axes[1].set_ylabel(f"joint {dim_y}")
    axes[1].set_title(
        f"{title_prefix}Sample {sample_idx} - path_pred\n"
        f"(dist_pred < {near_thresh})"
    )

    legend_right = [
        mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                      label='collision (0)', markersize=5),
        mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                      label='free non-path_pred', markersize=5),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                      label='path_pred (free & dist_pred<th)', markersize=5),
    ]
    axes[1].legend(handles=legend_right, loc='best', fontsize=8)

    plt.tight_layout()

    if env_idx is not None:
        out_name = f"single_ispath_{sample_idx:04d}_env_{env_idx:04d}.png"
    else:
        out_name = f"single_ispath_{sample_idx:04d}.png"

    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path)
    plt.close(fig)

# ================================================================
# main eval
# ================================================================
def main(args):
    # -------------------------
    # Logging
    # -------------------------
    model_name = "liche_pointnet"
    experiment_dir = os.path.join("results/model_training", model_name)
    log_dir = os.path.join(experiment_dir, "logs")
    fig_dir = os.path.join(experiment_dir, "figures", "test_vis")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    logger = logging.getLogger("ArmPlanningEval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, "eval_test.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(s):
        print(s)
        logger.info(s)

    log_string("========== EVAL TEST ==========")
    log_string(str(args))
    log_string(f"Experiment dir: {experiment_dir}")
    log_string(f"Checkpoint: {args.ckpt_path}")
    log_string(f"Eval mode: {args.eval_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Dataset / Dataloader (test)
    # -------------------------
    test_dataset = ArmPointCloudDataset(
        npz_path="data/liche/test/test.npz",
        max_points=None,
        shuffle_points=False,
        device=None,
    )

    # 如果设置了 eval_num_env，则随机采样这么多 env 来做评估
    if args.eval_num_env is not None:
        num_env = min(args.eval_num_env, len(test_dataset))
        sampler = RandomSampler(
            test_dataset,
            num_samples=num_env,
            replacement=False
        )
        log_string(f"[Eval] Randomly sample {num_env} envs from test set for evaluation.")
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_with_meta,
        )
    else:
        # 原行为：使用全部 test set，顺序不打乱
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn_with_meta,
        )

    # -------------------------
    # Model  （⚠️ 要和训练时超参一致）
    # -------------------------
    joint_in_dim = test_dataset.dof

    model = JointPointNetEncoder(
        joint_in_dim=joint_in_dim,
        joint_feat_dim=48,          # 与 train.py 保持一致
        env_latent_dim=60,          # train 时也是 60
        pointnet_embed_dim=128,     # 轻量版
        self_attn_layers=1,
        num_classes=3,
        self_attn_dropout=0.1,
        point_mlp_dropout=0.1,
        point_feat_dropout=0.1,
        cls_dropout=0.1,
    ).to(device)

    # 打印模型结构和参数量
    log_string("========== MODEL STRUCTURE ==========")
    log_string(str(model))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_string("========== MODEL PARAMETERS ==========")
    log_string(f"Total params        : {total_params:,}")
    log_string(f"Trainable params    : {trainable_params:,}")
    log_string(f"Non-trainable params: {total_params - trainable_params:,}")
    log_string("======================================")

    # 按模块统计参数量
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    log_string("========== MODEL PARAMS BY MODULE ==========")
    log_string(f"env_encoder params    : {count_parameters(model.env_encoder):,}")
    log_string(f"joint_encoder params  : {count_parameters(model.joint_encoder):,}")
    log_string(f"pointnet params       : {count_parameters(model.pointnet):,}")
    log_string(f"classifier mlp params : {count_parameters(model.mlp):,}")
    log_string("============================================")

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log_string(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")

    # ======================================================
    # 【单任务】单个环境延迟 benchmark（随机环境 + 随机关节）
    # ======================================================
    if args.eval_mode in ["single", "both"]:
        log_string("========== [SINGLE-TASK] 开始单环境延迟 benchmark (ispath by distance) ==========")
        single_vis_dir = os.path.join(
            experiment_dir, "figures", "single_task_ispath"
        )
        benchmark_single_sample(
            model,
            test_dataset,
            device,
            log_fn=log_string,
            warmup=10,
            num_run=50,
            num_env=1,
            max_points=2000,
            near_thresh=args.near_thresh,   # argparse 里要有这个参数
            vis_out_dir=single_vis_dir,
        )


    # 如果只想跑单任务，后面多任务评估直接跳过
    if args.eval_mode == "single":
        log_string("[INFO] eval_mode=single，只运行单任务 benchmark，不执行多任务全测试集评估。")
        return

    # ======================================================
    # 【多任务】全 test 集评估 + 混淆矩阵 + 推理时间统计
    # ======================================================

    # Loss (可选，如果想看 test loss) —— 和 train.py 对齐
    class_weights = torch.tensor([1.5, 1.0, 2.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # -------------------------
    # Eval on test set
    # -------------------------
    model.eval()
    num_classes = 3
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    total_loss = 0.0
    correct, total = 0, 0

    # ⏱️ 推理时间统计（拆成 CNN 和 head）
    total_cnn_time = 0.0      # 所有 batch CNN 时间总和（秒）
    total_head_time = 0.0     # 所有 batch head 时间总和（秒）
    total_points = 0          # 所有 batch 内点的总数
    total_envs = 0            # env 数

    # 用于可视化的 sample 计数
    vis_count = 0
    global_sample_idx = 0  # 跨 batch 的 sample 索引
    env_indices_order = list(test_loader.sampler)

    # 随机选择要可视化的 env（按 env 的 dataset 索引来）
    num_env_vis = min(args.num_vis, len(env_indices_order))
    vis_env_indices = set(
        np.random.choice(env_indices_order, size=num_env_vis, replace=False)
    )

    env_cursor = 0

    log_string("========== [MULTI-TASK] 开始全 test 集批量评估 ==========")

    with torch.no_grad():
        for env_voxel, joint_states, labels, meta in tqdm(
            test_loader, desc="[Test]"
        ):
            env_voxel = env_voxel.to(device)        # (B,1,D,H,W)
            joint_states = joint_states.to(device)  # (B,N,dof)
            labels = labels.to(device)              # (B,N)

            # 在 eval 中对关节点做随机子采样（N 维）
            if args.max_points_eval is not None:
                N_full = joint_states.shape[1]
                if N_full > args.max_points_eval:
                    perm = torch.randperm(N_full, device=device)[:args.max_points_eval]
                    joint_states = joint_states[:, perm, :]   # (B, M, dof)
                    labels = labels[:, perm]                  # (B, M)

            B, N, _ = joint_states.shape

            # 1）CNN：每个 env 只算一次 env_feat（按 batch 一起算）
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            env_feat = model.encode_env(env_voxel)           # (B, F_env)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            cnn_dt = t1 - t0

            # 2）head：只用 env_feat + joint_states
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            logits, _, _ = model.forward_with_env_feat(env_feat, joint_states)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.time()
            head_dt = t3 - t2

            total_cnn_time += cnn_dt
            total_head_time += head_dt
            total_points += B * N
            total_envs += B

            B, N, C = logits.shape

            loss = criterion(
                logits.view(B * N, C),
                labels.view(B * N)
            )
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)  # (B,N)
            mask = labels >= 0

            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

            # 更新混淆矩阵
            t = labels[mask].view(-1)
            p = preds[mask].view(-1)
            k = (t * num_classes + p).long()
            bincount = torch.bincount(k, minlength=num_classes * num_classes)
            confmat += bincount.view(num_classes, num_classes)

            # ✅ 可视化：随机 env，而不是前几个 env
            for b in range(B):
                if env_cursor + b < len(env_indices_order):
                    env_idx = env_indices_order[env_cursor + b]
                else:
                    env_idx = None

                if (env_idx is None) or (env_idx not in vis_env_indices):
                    global_sample_idx += 1
                    continue

                if vis_count >= args.num_vis:
                    global_sample_idx += 1
                    continue

                js_b = joint_states[b]  # (N,dof)
                lb_b = labels[b]        # (N,)
                pr_b = preds[b]         # (N,)

                visualize_sample_2d_joint(
                    js_b.cpu(), lb_b.cpu(), pr_b.cpu(),
                    out_dir=fig_dir,
                    sample_idx=global_sample_idx,
                    max_points_vis=args.max_points_vis,
                    env_idx=env_idx,
                )

                vis_count += 1
                global_sample_idx += 1
                vis_env_indices.remove(env_idx)

            env_cursor += B

    avg_test_loss = total_loss / len(test_loader)
    test_acc = correct / total if total > 0 else 0.0

    # -------------------------
    # Metrics from confusion matrix
    # -------------------------
    metrics = compute_metrics_from_confmat(confmat)
    per_prec = metrics["per_class"]["precision"]
    per_rec = metrics["per_class"]["recall"]
    per_iou = metrics["per_class"]["iou"]
    per_f1 = metrics["per_class"]["f1"]
    macro = metrics["macro"]

    macro_prec = macro["precision"].item()
    macro_rec = macro["recall"].item()
    macro_iou = macro["iou"].item()
    macro_f1 = macro["f1"].item()

    coll_prec = per_prec[0].item()
    coll_rec = per_rec[0].item()
    coll_iou = per_iou[0].item()
    coll_f1 = per_f1[0].item()

    free_prec = per_prec[1].item()
    free_rec = per_rec[1].item()
    free_iou = per_iou[1].item()
    free_f1 = per_f1[1].item()

    path_prec = per_prec[2].item()
    path_rec = per_rec[2].item()
    path_iou = per_iou[2].item()
    path_f1 = per_f1[2].item()

    log_string("========== TEST RESULT ==========")
    log_string(
        f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, "
        f"Macro IoU: {macro_iou:.4f}, Macro F1: {macro_f1:.4f}"
    )
    log_string(
        f"  Class 0 (collision) --- "
        f"Prec={coll_prec:.4f}, Rec={coll_rec:.4f}, "
        f"F1={coll_f1:.4f}, IoU={coll_iou:.4f}"
    )
    log_string(
        f"  Class 1 (free)      --- "
        f"Prec={free_prec:.4f}, Rec={free_rec:.4f}, "
        f"F1={free_f1:.4f}, IoU={free_iou:.4f}"
    )
    log_string(
        f"  Class 2 (path)      --- "
        f"Prec={path_prec:.4f}, Rec={path_rec:.4f}, "
        f"F1={path_f1:.4f}, IoU={path_iou:.4f}"
    )

    # -------------------------
    # ⏱️ 推理时间统计结果（分 CNN / head）
    # -------------------------
    if total_envs > 0 and total_points > 0:
        avg_cnn_env = total_cnn_time / total_envs
        avg_head_env = total_head_time / total_envs
        avg_total_env = avg_cnn_env + avg_head_env

        avg_total_point = (total_cnn_time + total_head_time) / total_points

        log_string("========== [MULTI-TASK] INFERENCE SPEED (split CNN/head) ==========")
        log_string(f"Avg CNN time per env   : {avg_cnn_env * 1000:.3f} ms/env")
        log_string(f"Avg head time per env  : {avg_head_env * 1000:.3f} ms/env")
        log_string(f"Avg total time per env : {avg_total_env * 1000:.3f} ms/env")
        log_string(f"Avg total per-point    : {avg_total_point * 1e6:.3f} us/point")
        est_2000_time = avg_total_point * 2000
        log_string(f"Estimated 2000-point sample (CNN+head): {est_2000_time * 1000:.3f} ms")
        log_string("======================================================")

    log_string(f"2D joint-space visualizations saved to: {fig_dir}")


# ================================================================
# run
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)
