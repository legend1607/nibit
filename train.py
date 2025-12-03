import os
import torch
import argparse
import logging
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ArmPointCloudDataset, collate_fn_with_meta
from model.encoders.joint_pointlite_encoder import JointPointNetEncoder

from tqdm import tqdm
import torch.nn.functional as F

# ================================================================
# argparse 参数解析
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser("ArmPlanningModel")

    # 数据 / 模型
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)

    args = parser.parse_args()
    return args


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
    recall    = TP / (TP + FN + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "per_class": {
            "precision": precision,  # shape: (C,)
            "recall":    recall,
            "iou":       iou,
            "f1":        f1,
        },
        "macro": {
            "precision": precision.mean(),
            "recall":    recall.mean(),
            "iou":       iou.mean(),
            "f1":        f1.mean(),
        }
    }
    return metrics


# ================================================================
# main
# ================================================================
def main(args):
    # -------------------------
    # Logging 设置
    # -------------------------
    model_name = f"liche_pointnet"
    experiment_dir = os.path.join('results/model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    tensor_dir = os.path.join(experiment_dir, 'tensorboard')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)

    logger = logging.getLogger("ArmPlanning")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{model_name}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(s):
        print(s)
        logger.info(s)

    log_string("========== PARAMETER ==========")
    log_string(str(args))
    log_string(f"Saving to: {experiment_dir}")

    # -------------------------
    # 随机种子（如有需要可固定）
    # -------------------------
    # torch.manual_seed(0)
    # np.random.seed(0)

    # -------------------------
    # Dataset
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ArmPointCloudDataset(
        npz_path="data/liche/train/train.npz",
        max_points=None,
        shuffle_points=True,
        device=None,
    )
    val_dataset = ArmPointCloudDataset(
        npz_path="data/liche/val/val.npz",
        max_points=None,
        shuffle_points=False,
        device=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_meta,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_with_meta,
    )

    # -------------------------
    # 模型
    # -------------------------
    joint_in_dim = train_dataset.dof  # e.g., 6 或 7

    model = JointPointNetEncoder(
        joint_in_dim=joint_in_dim,
        joint_feat_dim=64,
        env_latent_dim=32,
        pointnet_embed_dim=256,
        num_classes=3,
        dropout_p=0,
    ).to(device)

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.lr_decay
    )

    # Loss（这里给 path 更高的权重）
    class_weights = torch.tensor([1.5, 1.0, 3.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=tensor_dir)

    num_epochs = args.epoch
    num_classes = 3

    # 用于保存“最优模型”的指标（综合考虑 collision + path）
    best_score = -1.0

    # ============================================================
    # Train Loop
    # ============================================================
    for epoch in range(num_epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        total_loss = 0.0

        for env_voxel, joint_states, labels, meta in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            env_voxel = env_voxel.to(device)      # (B, 1, D, H, W)
            joint_states = joint_states.to(device)  # (B, N, D_joint)
            labels = labels.to(device)            # (B, N)

            logits, global_feat, local_feat = model(env_voxel, joint_states)
            B, N, C = logits.shape

            loss = criterion(
                logits.view(B * N, C),
                labels.view(B * N),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        log_string(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        tb_writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0

        confmat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

        with torch.no_grad():
            for env_voxel, joint_states, labels, meta in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
            ):
                env_voxel = env_voxel.to(device)
                joint_states = joint_states.to(device)
                labels = labels.to(device)

                logits, _, _ = model(env_voxel, joint_states)
                B, N, C = logits.shape

                val_loss = criterion(
                    logits.view(B * N, C),
                    labels.view(B * N)
                )
                total_val_loss += val_loss.item()

                preds = logits.argmax(dim=-1)  # (B, N)
                mask = labels >= 0             # 有效标签（这里其实全有效）

                correct += (preds[mask] == labels[mask]).sum().item()
                total   += mask.sum().item()

                # Confusion matrix
                t = labels[mask].view(-1)
                p = preds[mask].view(-1)
                k = (t * num_classes + p).long()
                bincount = torch.bincount(k, minlength=num_classes * num_classes)
                confmat += bincount.view(num_classes, num_classes)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        # ---- 从混淆矩阵计算 metrics ----
        metrics = compute_metrics_from_confmat(confmat)
        per_prec = metrics["per_class"]["precision"]
        per_rec  = metrics["per_class"]["recall"]
        per_iou  = metrics["per_class"]["iou"]
        per_f1   = metrics["per_class"]["f1"]
        macro    = metrics["macro"]

        macro_prec = macro["precision"].item()
        macro_rec  = macro["recall"].item()
        macro_iou  = macro["iou"].item()
        macro_f1   = macro["f1"].item()

        # 类 0 = collision, 类 2 = path
        coll_prec = per_prec[0].item()
        coll_rec  = per_rec[0].item()
        coll_iou  = per_iou[0].item()
        coll_f1   = per_f1[0].item()

        path_prec = per_prec[2].item()
        path_rec  = per_rec[2].item()
        path_iou  = per_iou[2].item()
        path_f1   = per_f1[2].item()

        # ---- 综合评分：collision F1 + path F1 的加权和 ----
        # 可以按任务重要性调整权重，例如 0.7/0.3
        alpha = 0.5  # collision 权重
        beta  = 0.5  # path 权重
        score = alpha * coll_f1 + beta * path_f1

        # ---- 打印日志 ----
        log_string(
            f"[Epoch {epoch+1}] "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Macro IoU: {macro_iou:.4f}, Macro F1: {macro_f1:.4f}, "
            f"Score(c0&c2 F1): {score:.4f}"
        )
        log_string(
            f"  Class 0 (collision) --- "
            f"Prec={coll_prec:.4f}, Rec={coll_rec:.4f}, "
            f"F1={coll_f1:.4f}, IoU={coll_iou:.4f}"
        )
        log_string(
            f"  Class 1 (free)      --- "
            f"Prec={per_prec[1].item():.4f}, Rec={per_rec[1].item():.4f}, "
            f"F1={per_f1[1].item():.4f}, IoU={per_iou[1].item():.4f}"
        )
        log_string(
            f"  Class 2 (path)      --- "
            f"Prec={path_prec:.4f}, Rec={path_rec:.4f}, "
            f"F1={path_f1:.4f}, IoU={path_iou:.4f}"
        )

        # ---- TensorBoard ----
        tb_writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        tb_writer.add_scalar("Acc/val",  val_acc,      epoch + 1)

        tb_writer.add_scalar("Macro/IoU",       macro_iou,  epoch + 1)
        tb_writer.add_scalar("Macro/F1",        macro_f1,   epoch + 1)
        tb_writer.add_scalar("Macro/Precision", macro_prec, epoch + 1)
        tb_writer.add_scalar("Macro/Recall",    macro_rec,  epoch + 1)

        # Collision 类（0）
        tb_writer.add_scalar("Collision/IoU",       coll_iou,  epoch + 1)
        tb_writer.add_scalar("Collision/F1",        coll_f1,   epoch + 1)
        tb_writer.add_scalar("Collision/Precision", coll_prec, epoch + 1)
        tb_writer.add_scalar("Collision/Recall",    coll_rec,  epoch + 1)

        # Path 类（2）
        tb_writer.add_scalar("Path/IoU",       path_iou,  epoch + 1)
        tb_writer.add_scalar("Path/F1",        path_f1,   epoch + 1)
        tb_writer.add_scalar("Path/Precision", path_prec, epoch + 1)
        tb_writer.add_scalar("Path/Recall",    path_rec,  epoch + 1)

        # 综合 Score
        tb_writer.add_scalar("Score/collision_path_F1", score, epoch + 1)

        # ---- 保存“最优模型”：按 score（collision F1 + path F1） ----
        if score > best_score:
            best_score = score

            ckpt_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "macro_iou": macro_iou,
                    "macro_f1": macro_f1,
                    "collision_f1": coll_f1,
                    "path_f1": path_f1,
                    "score": best_score,
                },
                ckpt_path,
            )
            log_string(
                f"⭐ New best model saved at epoch {epoch+1}: "
                f"Score={best_score:.4f}, "
                f"Collision F1={coll_f1:.4f}, Path F1={path_f1:.4f}"
            )

        # ---- 更新学习率 ----
        scheduler.step()

    tb_writer.close()
    log_string("Training finished!!")


# ================================================================
# Run
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)
