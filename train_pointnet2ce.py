import os
import logging
import argparse
import importlib
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # ✅ 新增
from model.PathPlanDataLoader import PathPlanDataset


# ======================
# 参数设置
# ======================
def parse_args():
    parser = argparse.ArgumentParser('Train PointNet++ DualHead with TensorBoard')
    parser.add_argument('--dim', type=int, default=2, help='Environment dimension (2 or 3)')
    parser.add_argument('--model', type=str, default='pointnet2_dualhead_attention', help='Model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', type=int, default=200, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer (Adam or SGD)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=20, help='LR decay step')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='LR decay rate')
    parser.add_argument('--npoint', type=int, default=2048)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--wp', type=float, default=0.7, help='Weight for path IoU')
    parser.add_argument('--wk', type=float, default=0.3, help='Weight for keypoint IoU')
    return parser.parse_args()


# ======================
# 主训练函数
# ======================
def main(args):
    # ---- 文件和日志目录 ----
    model_name = args.model + f"_{args.dim}d"
    experiment_dir = os.path.join('results', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # ---- logging ----
    logging.basicConfig(
        filename=os.path.join(experiment_dir, f'{model_name}.log'),
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    log = lambda s: (print(s), logging.info(s))

    # ---- TensorBoard ---- ✅
    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "tensorboard"))

    # ---- 随机种子 ----
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    log(f"Training {model_name} with seed {args.random_seed}")

    # ---- 数据集 ----
    env_type = f"random_{args.dim}d"
    train_ds = PathPlanDataset(env_type, f"data/{env_type}/train.npz")
    val_ds = PathPlanDataset(env_type, f"data/{env_type}/val.npz")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    log(f"Loaded train={len(train_ds)}, val={len(val_ds)}")

    # ---- 模型 ----
    MODEL = importlib.import_module(f"model.{args.model}")
    model = MODEL.get_model(coord_dim=args.dim, feature_dim=3).cuda()
    criterion = MODEL.get_loss_dualhead().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

    best_score = None
    global_step = 0

    # ======================
    # 训练循环
    # ======================
    for epoch in range(args.epoch):
        log(f"\nEpoch [{epoch+1}/{args.epoch}] --------------------------")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            pc_raw, pc_xyz, pc_features, path_mask, key_mask, token = batch

            pc_xyz = torch.tensor(pc_xyz, dtype=torch.float32)
            points = torch.cat([pc_xyz, pc_features], dim=2).transpose(2, 1).cuda()
            path_mask, key_mask = path_mask.cuda(), key_mask.cuda()

            optimizer.zero_grad()
            path_logits, key_logits = model(points)
            loss_total, loss_path, loss_key = criterion(path_logits, key_logits, path_mask, key_mask)
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()
            global_step += 1

            # ✅ TensorBoard 每个 step 记录训练 loss
            writer.add_scalar('Train/BatchLoss', loss_total.item(), global_step)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
        log(f"Train Loss: {avg_train_loss:.6f}")

        # ======================
        # 验证阶段
        # ======================
        model.eval()
        val_loss, iou_path_sum, iou_key_sum, total_seen = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                pc_raw, pc_xyz, pc_features, path_mask, key_mask, token = batch
                points = torch.cat([pc_xyz, pc_features], dim=2).transpose(2, 1).cuda()
                path_mask, key_mask = path_mask.cuda(), key_mask.cuda()

                path_logits, key_logits = model(points)
                loss_total, _, _ = criterion(path_logits, key_logits, path_mask, key_mask)
                val_loss += loss_total.item()
                total_seen += 1

                # IoU 计算
                path_pred = (torch.sigmoid(path_logits) > 0.5).float()
                key_pred = (torch.sigmoid(key_logits) > 0.5).float()

                inter_path = (path_pred * path_mask).sum(dim=[1, 2])
                union_path = ((path_pred + path_mask) > 0).float().sum(dim=[1, 2])
                inter_key = (key_pred * key_mask).sum(dim=[1, 2])
                union_key = ((key_pred + key_mask) > 0).float().sum(dim=[1, 2])

                iou_path_sum += (inter_path / (union_path + 1e-6)).mean().item()
                iou_key_sum += (inter_key / (union_key + 1e-6)).mean().item()

        mean_iou_path = iou_path_sum / total_seen
        mean_iou_key = iou_key_sum / total_seen
        avg_val_loss = val_loss / total_seen
        score = args.wp * mean_iou_path + args.wk * mean_iou_key

        log(f"[Val] Loss={avg_val_loss:.6f}  PathIoU={mean_iou_path:.4f}  KeyIoU={mean_iou_key:.4f}  Score={score:.4f}")

        # ✅ TensorBoard 记录验证指标
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Path_IoU', mean_iou_path, epoch)
        writer.add_scalar('Val/Key_IoU', mean_iou_key, epoch)
        writer.add_scalar('Val/Weighted_Score', score, epoch)

        # ======================
        # 保存最优模型
        # ======================
        if best_score is None or score >= best_score:
            best_score = score
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'path_iou': mean_iou_path,
                'key_iou': mean_iou_key,
                'score': score
            }
            ckpt_path = os.path.join(checkpoints_dir, f'best_{model_name}.pth')
            torch.save(save_dict, ckpt_path)
            log(f"✅ Saved best model to {ckpt_path} (score={score:.4f})")

    writer.close()
    log(f"Training complete. Best weighted score = {best_score:.4f}")


# ======================
# 运行入口
# ======================
if __name__ == '__main__':
    args = parse_args()
    main(args)
