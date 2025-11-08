# train_pointnet2tf.py
import os
import time
import argparse
import logging
import importlib
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model.PathPlanDataLoader import PathPlanDataset  # 注意你最后贴的 DataLoader 路径

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def bn_momentum_adjust(m, momentum):
    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        m.momentum = momentum

def parse_args():
    parser = argparse.ArgumentParser('Train PointNet2TF (path + keypoint [+ direction])')
    parser.add_argument('--env', type=str, default='random', choices=['random', 'kuka_random'])
    parser.add_argument('--dim', type=int, default=2, help='environment dimension: 2 or 3')
    parser.add_argument('--planner_type', type=str, default='bitstar', choices=['bitstar', 'rrt_star'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--use_direction', action='store_true', help='Enable direction head and loss')
    parser.add_argument('--freeze_direction_epochs', type=int, default=50,
                        help='Number of initial epochs to freeze direction head (default 0)')
    parser.add_argument('--save_by', type=str, default='combined', choices=['combined', 'path_iou', 'keypoint_f1'],
                        help='Which metric to use for saving best model')
    return parser.parse_args()

# ---------- metrics helpers ----------
def batch_path_iou(path_logits, path_label):
    # path_logits: [B,1,N], path_label: [B,N]
    prob = torch.sigmoid(path_logits.squeeze(1))
    pred = (prob > 0.5).float()
    gt = (path_label > 0.5).float()
    inter = (pred * gt).sum().item()
    union = ((pred + gt) > 0).float().sum().item()
    return inter, union

def batch_keypoint_f1(key_logits, key_label):
    # key_logits: [B,1,N], key_label: [B,N]
    prob = torch.sigmoid(key_logits.squeeze(1))
    pred = (prob > 0.5).float()
    gt = (key_label > 0.5).float()
    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    return tp, fp, fn

# ---------- main ----------
def main(args):
    # setup logging / dirs
    model_name = f"{args.env}_pointnet2tf_{args.dim}d_{args.planner_type}"
    print("model name:",model_name)
    experiment_dir = os.path.join('results', 'model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("TrainPointNet2TF")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, f'{model_name}.txt'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    def log(s):
        print(s)
        logger.info(s)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    log("Arguments: " + str(args))
    tb_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))

    # datasets / dataloaders
    env_type = f"{args.env}_{args.dim}d_{args.planner_type}"
    TRAIN_DATASET = PathPlanDataset(env_type, dataset_filepath=f"data/{env_type}/train.npz")
    VAL_DATASET = PathPlanDataset(env_type, dataset_filepath=f"data/{env_type}/val.npz")

    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=8, drop_last=False)
    val_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=8, drop_last=False)

    log(f"Train samples: {len(TRAIN_DATASET)}, Val samples: {len(VAL_DATASET)}, n_points={TRAIN_DATASET.n_points}")

    # import model
    MODEL = importlib.import_module('pointnet_pointnet2tf.models.pointnet2tf')
    classifier = MODEL.get_model(num_classes=1, coord_dim=TRAIN_DATASET.d, use_direction=args.use_direction).cuda()
    criterion = MODEL.get_loss(use_direction=args.use_direction).cuda()
    classifier.apply(inplace_relu)

    # weight init (optional)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            try:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            except:
                pass
        elif classname.find('Conv1d') != -1:
            try:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            except:
                pass
        elif classname.find('Linear') != -1:
            try:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            except:
                pass
    classifier.apply(weights_init)

    # optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate) \
        if args.optimizer == 'Adam' else torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # lr / bn schedule settings
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    # resume checkpoint if exists
    start_epoch = 0
    best_path_iou = None
    best_keypoint_f1 = None
    best_combined = None
    ckpt_path = os.path.join(checkpoints_dir, f'best_{model_name}.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        classifier.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch = ckpt.get('epoch', 0) + 1
        best_path_iou = ckpt.get('best_path_iou', None)
        best_keypoint_f1 = ckpt.get('best_keypoint_f1', None)
        best_combined = ckpt.get('best_combined', None)
        log(f"Loaded checkpoint {ckpt_path}, resume from epoch {start_epoch}")

    global_epoch = start_epoch
    # ---------- 初始冻结（在训练开始前一次性做） ----------
    if args.use_direction and hasattr(classifier, 'direction_head') and args.freeze_direction_epochs > 0:
        log(f"Freezing direction head for first {args.freeze_direction_epochs} epochs...")
        for p in classifier.direction_head.parameters():
            p.requires_grad = False

    for epoch in range(start_epoch, args.epoch):
        log(f"==== Epoch {epoch+1}/{args.epoch} ====")

        # lr update
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        log(f"LR: {lr:.6f}")
        # ---------- 在到达解冻点时一次性解冻 ----------
        if epoch == args.freeze_direction_epochs:
            log("Unfreezing direction head and resetting optimizer")
            for p in classifier.direction_head.parameters():
                p.requires_grad = True

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, classifier.parameters()),
                lr=lr, weight_decay=args.decay_rate
            )

        # ---------- 日志：打印当前冻结状态 ----------
        if hasattr(classifier, 'direction_head'):
            grad_status = [p.requires_grad for p in classifier.direction_head.parameters()]
            log(f"[Epoch {epoch+1}] Direction head grad enabled = {any(grad_status)}")


        # BN momentum adjust
        momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)), 0.01)
        classifier.apply(lambda m: bn_momentum_adjust(m, momentum))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ============== training ==============
        classifier.train()
        train_total_loss = 0.0
        train_path_loss = 0.0
        train_key_loss = 0.0
        train_dir_loss = 0.0
        nbatches = 0
        for batch in tqdm(train_loader, total=len(train_loader), smoothing=0.9, desc='Train'):
            pc_pos_raw, pc_pos, pc_features, path_mask, direction_labels, keypoint_labels, token = batch

            # ===== 数据增强 =====
            pc_xyz_np = pc_pos if isinstance(pc_pos, np.ndarray) else pc_pos.cpu().numpy()

            # --- 确认 pc_pos 为 numpy (B,N,C) 用于 augmentation ---
            # if isinstance(pc_pos, torch.Tensor):
            #     pc_pos_np = pc_pos.cpu().numpy()
            # else:
            #     pc_pos_np = pc_pos

            # # --- 数据增强（返回 rot_mats 用于同步方向） ---
            # if args.env.startswith('kuka') and env is not None:
            #     pc_xyz_np = point_utils.augment_kuka_joint_space(pc_pos_np, env=env)
            #     rot_mats = None
            # else:
            #     pc_xyz_np, rot_mats = point_utils.rotate_point_cloud_z(pc_pos_np, return_rotmat=True)

            #     # 同步旋转 direction_labels （only if use_direction）
            #     if args.use_direction and direction_labels is not None:
            #         dir_np = direction_labels.cpu().numpy()  # [B, N, 2]
            #         rotated_dirs = np.matmul(dir_np, rot_mats.transpose(0, 2, 1))  # [B, N, 2]
            #         direction_labels = torch.from_numpy(rotated_dirs).float()

            # --- 转回 tensor，移动到 device（non_blocking 以配合 pin_memory） ---
            pc_xyz = torch.from_numpy(pc_xyz_np).float().to(device, non_blocking=True)   # [B, N, C]
            pc_features = pc_features.float().to(device, non_blocking=True)             # [B, N, feat]
            points = torch.cat([pc_xyz, pc_features], dim=2).transpose(2, 1).contiguous()  # [B, C, N]

            # --- 标签到 device ---
            path_label = path_mask.float().to(device, non_blocking=True)
            keypoint_label = keypoint_labels.float().to(device, non_blocking=True)
            direction_label = direction_labels.float().to(device, non_blocking=True) if args.use_direction else None

            # forward
            path_pred, keypoint_pred, direction_pred = classifier(points)  # path: [B,1,N], key: [B,1,N], dir: [B,d,N] or None
            total_loss, p_loss, k_loss, d_loss = criterion(path_pred, keypoint_pred, direction_pred,
                                                           path_label, keypoint_label, direction_label)

            optimizer.zero_grad()
            total_loss.backward()
            # === Freeze-check: ensure direction head grads are zero during freeze ===
            # if epoch < args.freeze_direction_epochs and hasattr(classifier, 'direction_head'):
            #     for name, param in classifier.direction_head.named_parameters():
            #         if param.grad is not None:
            #             # 使用范数检查是否接近 0（避免浮点误差问题）
            #             max_abs = param.grad.detach().abs().max().item()
            #             assert max_abs < 1e-9, f"[Freeze check fail] {name} has non-zero grad ({max_abs}) during freeze!"
            # if hasattr(classifier, 'direction_head'):
            #     first_param = next(classifier.direction_head.parameters())
            #     log(f"[DEBUG] Direction head param mean={first_param.data.mean().item():.6e}")
            #     grads = [p.grad.abs().mean().item() if p.grad is not None else 0.0 for p in classifier.direction_head.parameters()]
            #     log(f"[DEBUG] Direction head grad mean={np.mean(grads):.6e}")

            optimizer.step()

            train_total_loss += total_loss.item()
            train_path_loss += p_loss.item()
            train_key_loss += k_loss.item()
            # d_loss may be tensor scalar
            train_dir_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else float(d_loss)
            nbatches += 1

        train_total_loss /= (nbatches + 1e-12)
        train_path_loss /= (nbatches + 1e-12)
        train_key_loss /= (nbatches + 1e-12)
        train_dir_loss /= (nbatches + 1e-12)

        log(f"Train Loss: total={train_total_loss:.6f}, path={train_path_loss:.6f}, key={train_key_loss:.6f}, dir={train_dir_loss:.6f}")
        tb_writer.add_scalar('Train/TotalLoss', train_total_loss, global_epoch)
        tb_writer.add_scalar('Train/PathLoss', train_path_loss, global_epoch)
        tb_writer.add_scalar('Train/KeyLoss', train_key_loss, global_epoch)
        tb_writer.add_scalar('Train/DirLoss', train_dir_loss, global_epoch)

        # ============== validation ==============
        classifier.eval()
        with torch.no_grad():
            val_total_loss = 0.0
            val_path_loss = 0.0
            val_key_loss = 0.0
            val_dir_loss = 0.0
            nval = 0

            inter_sum = 0.0
            union_sum = 0.0
            tp_sum = 0.0
            fp_sum = 0.0
            fn_sum = 0.0
            direction_cos_sum = 0.0
            direction_ang_sum = 0.0
            valid_dir_batches = 0

            for batch in tqdm(val_loader, total=len(val_loader), smoothing=0.9, desc='Val'):
                pc_pos_raw, pc_pos, pc_features, path_mask, direction_labels, keypoint_labels, token = batch

                # assume val no augmentation; if you want same augment, reuse train augmentation
                if isinstance(pc_pos, torch.Tensor):
                    pc_xyz = pc_pos.float().to(device, non_blocking=True)
                else:
                    pc_xyz = torch.from_numpy(pc_pos).float().to(device, non_blocking=True)

                pc_features_t = pc_features.float().to(device, non_blocking=True)
                points = torch.cat([pc_xyz, pc_features_t], dim=2).transpose(2,1).contiguous()

                path_label = path_mask.float().to(device, non_blocking=True)
                keypoint_label = keypoint_labels.float().to(device, non_blocking=True)
                direction_label = direction_labels.float().to(device, non_blocking=True) if args.use_direction else None

                path_pred, keypoint_pred, direction_pred = classifier(points)
                total_loss, p_loss, k_loss, d_loss = criterion(path_pred, keypoint_pred, direction_pred,
                                                               path_label, keypoint_label, direction_label)

                val_total_loss += total_loss.item()
                val_path_loss += p_loss.item()
                val_key_loss += k_loss.item()
                val_dir_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else float(d_loss)
                nval += 1
                if args.use_direction and direction_label is not None and direction_pred is not None:
                    # mask 有效方向点
                    mask = torch.norm(direction_label, dim=1) > 1e-3
                    if mask.sum() > 0:
                        # 归一化向量
                        pred_norm = F.normalize(direction_pred, dim=2) 
                        gt_norm   = F.normalize(direction_label.permute(0,2,1), dim=2)
                        cos_sim = (pred_norm[mask] * gt_norm[mask]).sum(dim=-1)
                        cos_sim = torch.clamp(cos_sim, -1, 1)
                        ang_err = torch.acos(cos_sim) * 180 / np.pi  # 转为角度
                        mean_ang_err = ang_err.mean().item()
                        mean_cos_sim = cos_sim.mean().item()

                        direction_cos_sum += mean_cos_sim
                        direction_ang_sum += mean_ang_err
                        valid_dir_batches += 1

                inter, union = batch_path_iou(path_pred, path_label)
                inter_sum += inter
                union_sum += union

                tp, fp, fn = batch_keypoint_f1(keypoint_pred, keypoint_label)
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

            val_total_loss /= (nval + 1e-12)
            val_path_loss /= (nval + 1e-12)
            val_key_loss /= (nval + 1e-12)
            val_dir_loss /= (nval + 1e-12)

            path_iou = inter_sum / (union_sum + 1e-12)
            precision = tp_sum / (tp_sum + fp_sum + 1e-12)
            recall = tp_sum / (tp_sum + fn_sum + 1e-12)
            keypoint_f1 = 2 * precision * recall / (precision + recall + 1e-12)
            

            log(f"Val Loss: total={val_total_loss:.6f}, path={val_path_loss:.6f}, key={val_key_loss:.6f}, dir={val_dir_loss:.6f}")
            log(f"Val Path IoU: {path_iou:.6f} | Keypoint F1: {keypoint_f1:.6f} (P={precision:.4f}, R={recall:.4f})")
            if args.use_direction and valid_dir_batches > 0:
                mean_dir_cos = direction_cos_sum / valid_dir_batches
                mean_dir_ang = direction_ang_sum / valid_dir_batches
                log(f"Val Direction: cos_sim={mean_dir_cos:.4f}, mean_ang_err={mean_dir_ang:.2f}°")
                tb_writer.add_scalar('Val/DirCosineSim', mean_dir_cos, global_epoch)
                tb_writer.add_scalar('Val/DirAngularErr', mean_dir_ang, global_epoch)
            else:
                mean_dir_cos, mean_dir_ang = 0.0, 180.0

            tb_writer.add_scalar('Val/TotalLoss', val_total_loss, global_epoch)
            tb_writer.add_scalar('Val/PathLoss', val_path_loss, global_epoch)
            tb_writer.add_scalar('Val/KeyLoss', val_key_loss, global_epoch)
            tb_writer.add_scalar('Val/DirLoss', val_dir_loss, global_epoch)
            tb_writer.add_scalar('Val/PathIoU', path_iou, global_epoch)
            tb_writer.add_scalar('Val/KeyF1', keypoint_f1, global_epoch)
            tb_writer.add_scalar('Val/KeyPrecision', precision, global_epoch)
            tb_writer.add_scalar('Val/KeyRecall', recall, global_epoch)
            tb_writer.add_scalar('Val/DirCosineSim', mean_dir_cos, global_epoch)
            tb_writer.add_scalar('Val/DirAngularErr', mean_dir_ang, global_epoch)

            # decide saving best model
            if args.save_by == 'path_iou':
                better = (best_path_iou is None) or (path_iou > best_path_iou)
            elif args.save_by == 'keypoint_f1':
                better = (best_keypoint_f1 is None) or (keypoint_f1 > best_keypoint_f1)
            else:  # combined
                if args.use_direction:
                    combined = 0.4 * path_iou + 0.3 * keypoint_f1 + 0.3 * mean_dir_cos
                else:
                    combined = 0.6 * path_iou + 0.4 * keypoint_f1
                better = (best_combined is None) or (combined > best_combined)

            if better:
                # update bests accordingly
                if args.save_by == 'path_iou':
                    best_path_iou = path_iou
                elif args.save_by == 'keypoint_f1':
                    best_keypoint_f1 = keypoint_f1
                else:
                    best_combined = combined

                savepath = os.path.join(checkpoints_dir, f'best_{model_name}.pth')
                log(f"Saving best model to {savepath} (mode={args.save_by}, use_dir={args.use_direction})")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_path_iou': best_path_iou,
                    'best_keypoint_f1': best_keypoint_f1,
                    'best_combined': best_combined,
                    'best_dir_cosine': mean_dir_cos,
                    'best_dir_ang': mean_dir_ang,
                }, savepath)

        global_epoch += 1

    tb_writer.close()
    log("Training finished.")

if __name__ == '__main__':
    args = parse_args()
    main(args)