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
import time  # ✅ 用于计时


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

    # ✅ 新增：eval 随机环境 & 随机关节
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

    return parser.parse_args()


# ================================================================
# 单样本 benchmark：随机环境 + 随机关节
# ================================================================
def benchmark_single_sample(model, dataset, device, log_fn=print,
                            warmup=10, num_run=50, num_env=5, max_points=None):
    """
    随机选 num_env 个环境；每个环境再随机选 max_points 个关节角度点做 benchmark。
    如果 max_points 为 None，则使用该环境里的所有点。
    """
    model.eval()
    import random
    indices = random.sample(range(len(dataset)), k=min(num_env, len(dataset)))

    times = []
    for idx in indices:
        sample = dataset[idx]
        env_voxel, joint_states, labels, meta = sample   # joint_states: (N, dof)

        # ---------- 随机关节角度子集 ----------
        N = joint_states.shape[0]
        if max_points is not None and N > max_points:
            perm = torch.randperm(N)[:max_points]
            joint_states = joint_states[perm]    # (max_points, dof)
            if labels is not None:
                labels = labels[perm]
        # -------------------------------------

        env_voxel = env_voxel.unsqueeze(0).to(device)        # (1, 1, D, H, W)
        joint_states = joint_states.unsqueeze(0).to(device)  # (1, M, dof) M=子采样后点数
        B, M, _ = joint_states.shape  # B=1

        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(env_voxel, joint_states)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(num_run):
                _ = model(env_voxel, joint_states)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        times.append((t1 - t0) / num_run)

    avg_time = sum(times) / len(times)
    per_point = avg_time / (B * M)

    log_fn("========== SINGLE SAMPLE BENCH (multi-env, random joints) ==========")
    log_fn(f"Indices              : {indices}")
    log_fn(f"Num points per env   : {M} (after random sampling)")
    log_fn(f"Avg forward time     : {avg_time * 1000:.3f} ms / env")
    log_fn(f"Avg per-point time   : {per_point * 1e6:.3f} us / point")
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

    # ✅ 防止有 ignore_index（例如 -1）
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

    # ✅ 文件名也带上 env_idx，方便回溯
    if env_idx is not None:
        out_name = f"sample_{sample_idx:04d}_env_{env_idx:04d}.png"
    else:
        out_name = f"sample_{sample_idx:04d}.png"

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

    # ✅ 如果设置了 eval_num_env，则随机采样这么多 env 来做评估
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
    # Model
    # -------------------------
    joint_in_dim = test_dataset.dof

    model = JointPointNetEncoder(
        joint_in_dim=joint_in_dim,
        joint_feat_dim=64,
        env_latent_dim=60,
        pointnet_embed_dim=256,
        num_classes=3,
        dropout_p=0,
    ).to(device)

    # ====== 打印模型结构和参数量 ======
    log_string("========== MODEL STRUCTURE ==========")
    log_string(str(model))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_string("========== MODEL PARAMETERS ==========")
    log_string(f"Total params       : {total_params:,}")
    log_string(f"Trainable params   : {trainable_params:,}")
    log_string(f"Non-trainable params: {total_params - trainable_params:,}")
    log_string("======================================")

    # 按模块统计参数量
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    log_string("========== MODEL PARAMS BY MODULE ==========")
    log_string(f"env_encoder params   : {count_parameters(model.env_encoder):,}")
    log_string(f"joint_encoder params : {count_parameters(model.joint_encoder):,}")
    log_string(f"pointnet params      : {count_parameters(model.pointnet):,}")
    log_string(f"classifier mlp params: {count_parameters(model.mlp):,}")
    log_string("============================================")

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log_string(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")

    # -------------------------
    # 单个环境延迟 benchmark（随机环境 + 随机关节）
    # -------------------------
    log_string("Running single-sample latency benchmark on random env & joints ...")
    benchmark_single_sample(
        model,
        test_dataset,
        device,
        log_fn=log_string,
        warmup=10,
        num_run=50,
        num_env=5,
        max_points=2000   # 随机抽 2000 个关节角度
    )

    # Loss (可选，如果想看 test loss)
    class_weights = torch.tensor([1.5, 1.0, 3.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # -------------------------
    # Eval on test set
    # -------------------------
    model.eval()
    num_classes = 3
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    total_loss = 0.0
    correct, total = 0, 0

    # ⏱️ 推理时间统计
    total_infer_time = 0.0     # 所有 batch 推理时间总和（秒）
    total_infer_points = 0     # 所有 batch 内点的总数
    total_samples = 0          # 所有样本数
    num_batches = 0

    # 用于可视化的 sample 计数
    vis_count = 0
    global_sample_idx = 0  # 跨 batch 的 sample 索引
    # ✅ 当前 DataLoader 中 env 的遍历顺序（对应 dataset 的 index）
    env_indices_order = list(test_loader.sampler)

    # ✅ 随机选择要可视化的 env（按 env 的 dataset 索引来）
    # 可视化的 env 数量不超过 args.num_vis
    num_env_vis = min(args.num_vis, len(env_indices_order))
    vis_env_indices = set(
        np.random.choice(env_indices_order, size=num_env_vis, replace=False)
    )

    env_cursor = 0
    with torch.no_grad():
        for env_voxel, joint_states, labels, meta in tqdm(
            test_loader, desc="[Test]"
        ):
            env_voxel = env_voxel.to(device)        # (B,1,D,H,W)
            joint_states = joint_states.to(device)  # (B,N,dof)
            labels = labels.to(device)              # (B,N)

            # ✅ 在 eval 中对关节点做随机子采样（N 维）
            if args.max_points_eval is not None:
                N_full = joint_states.shape[1]
                if N_full > args.max_points_eval:
                    perm = torch.randperm(N_full, device=device)[:args.max_points_eval]
                    joint_states = joint_states[:, perm, :]   # (B, M, dof)
                    labels = labels[:, perm]                  # (B, M)

            B, N, _ = joint_states.shape

            # -------------------------
            # ⏱️ 计时：一次 batch 推理
            # -------------------------
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            logits, _, _ = model(env_voxel, joint_states)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            dt = t1 - t0
            total_infer_time += dt
            total_infer_points += B * N
            total_samples += B
            num_batches += 1
            # -------------------------

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

            # -------------------------
            # ✅ 可视化：随机 env，而不是前几个 env
            # -------------------------
            for b in range(B):
                # 这个样本在 dataset 中对应的 env index
                if env_cursor + b < len(env_indices_order):
                    env_idx = env_indices_order[env_cursor + b]
                else:
                    env_idx = None  # 理论上不会发生，只是兜底

                # 如果这个 env 不在我们随机抽出来的列表里，就跳过
                if (env_idx is None) or (env_idx not in vis_env_indices):
                    global_sample_idx += 1
                    continue

                # 如果已经达到可视化上限，也可以直接 break
                if vis_count >= args.num_vis:
                    global_sample_idx += 1
                    continue

                # 拿出第 b 个样本
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
                # 这个 env 已经画过了，从集合里移除，避免重复
                vis_env_indices.remove(env_idx)

            # ✅ 无论是否可视化，这个 batch 的 B 个 env 都已经“消费掉”
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
    # ⏱️ 推理时间统计结果
    # -------------------------
    if num_batches > 0 and total_samples > 0 and total_infer_points > 0:
        avg_batch_time = total_infer_time / num_batches        # 秒 / batch
        avg_sample_time = total_infer_time / total_samples     # 秒 / sample
        avg_point_time = total_infer_time / total_infer_points # 秒 / point

        log_string("========== INFERENCE SPEED ==========")
        log_string(f"Avg batch infer time  : {avg_batch_time * 1000:.3f} ms/batch")
        log_string(f"Avg sample infer time : {avg_sample_time * 1000:.3f} ms/sample")
        log_string(f"Avg point infer time  : {avg_point_time * 1e6:.3f} us/point")
        # 如果典型是 2000 points，可以给一个直观估计
        est_2000_time = avg_point_time * 2000
        log_string(f"Estimated 2000-point sample: {est_2000_time * 1000:.3f} ms")
        log_string("======================================")

    log_string(f"2D joint-space visualizations saved to: {fig_dir}")


# ================================================================
# run
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)
