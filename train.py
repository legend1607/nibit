# train.py

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ArmPointCloudDataset
from model.encoders.joint_pointlite_encoder import JointPointNetEncoder

from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset & DataLoader ===
train_dataset = ArmPointCloudDataset(
    npz_path="data/liche/train/train.npz",
    max_points=None,          # 或者比如 2048
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
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# === 模型 ===
joint_in_dim = train_dataset.dof  # 一般是 7

model = JointPointNetEncoder(
    joint_in_dim=joint_in_dim,
    joint_feat_dim=64,
    env_latent_dim=60,       # CNN_3D 输出维度
    pointnet_embed_dim=256,
    num_classes=3,
    dropout_p=0,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 类别权重：路径点稍微加权
class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# === TensorBoard ===
log_dir = "runs/arm_pointnet_liche_iou"
writer = SummaryWriter(log_dir=log_dir)

# === IoU / Precision / Recall / F1 计算 ===
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
            "precision": precision.cpu().tolist(),
            "recall":    recall.cpu().tolist(),
            "iou":       iou.cpu().tolist(),
            "f1":        f1.cpu().tolist(),
        },
        "macro": {
            "precision": precision.mean().item(),
            "recall":    recall.mean().item(),
            "iou":       iou.mean().item(),
            "f1":        f1.mean().item(),
        }
    }
    return metrics

num_epochs = 50
num_classes = 3
best_val_acc = 0.0
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

for epoch in range(num_epochs):
    # ====== Train ======
    model.train()
    total_loss = 0.0

    for env_voxel, joint_states, labels, meta in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
    ):
        env_voxel = env_voxel.to(device, non_blocking=True)
        joint_states = joint_states.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, global_feat, local_feat = model(env_voxel, joint_states)  # (B,N,3)

        B, N, C = logits.shape
        loss = criterion(
            logits.view(B * N, C),
            labels.view(B * N)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)

    # ====== Eval ======
    model.eval()
    correct, total = 0, 0
    total_val_loss = 0.0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for env_voxel, joint_states, labels, meta in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
        ):
            env_voxel = env_voxel.to(device, non_blocking=True)
            joint_states = joint_states.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _, _ = model(env_voxel, joint_states)
            B, N, C = logits.shape

            val_loss = criterion(
                logits.view(B * N, C),
                labels.view(B * N)
            )
            total_val_loss += val_loss.item()

            preds = logits.argmax(dim=-1)  # (B,N)
            mask = labels >= 0
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

            # 更新混淆矩阵
            t = labels[mask].view(-1)
            p = preds[mask].view(-1)
            k = (t * num_classes + p).to(torch.long)
            bincount = torch.bincount(k, minlength=num_classes * num_classes)
            confmat += bincount.view(num_classes, num_classes)

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = correct / total if total > 0 else 0.0

    metrics = compute_metrics_from_confmat(confmat)
    per_prec = metrics["per_class"]["precision"]
    per_rec  = metrics["per_class"]["recall"]
    per_iou  = metrics["per_class"]["iou"]
    per_f1   = metrics["per_class"]["f1"]
    macro    = metrics["macro"]

    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"  Macro Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f}, "
          f"F1: {macro['f1']:.4f}, IoU: {macro['iou']:.4f}")

    for cls in range(num_classes):
        print(f"    Class {cls}: "
              f"Prec={per_prec[cls]:.4f}, "
              f"Rec={per_rec[cls]:.4f}, "
              f"F1={per_f1[cls]:.4f}, "
              f"IoU={per_iou[cls]:.4f}")

    print(f"  >>> Path class (2) --- "
          f"Prec={per_prec[2]:.4f}, Rec={per_rec[2]:.4f}, "
          f"F1={per_f1[2]:.4f}, IoU={per_iou[2]:.4f}")

    # ====== TensorBoard 记录关键指标 ======
    writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
    writer.add_scalar("Acc/val", val_acc, epoch + 1)
    writer.add_scalar("IoU/path", per_iou[2], epoch + 1)
    writer.add_scalar("F1/path", per_f1[2], epoch + 1)

    # ====== 保存 val_acc 最优模型（可选） ======
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "metrics": metrics,
            },
            ckpt_path,
        )
        print(f"⭐ New best model saved at epoch {epoch+1} with Val Acc={val_acc:.4f}")

# 训练结束
writer.close()
print("✅ Training finished.")
