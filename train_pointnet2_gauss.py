import os
import logging
import argparse
import importlib
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.PathPlanDataLoader import PathPlanDataset  # 自己实现的 Dataset

# -----------------------------
# 参数解析
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser('PointNet++ Path & Keypoint Training')
    parser.add_argument('--dim', type=int, default=2, help='environment dimension: 2 or 3')
    parser.add_argument('--model', type=str, default='pointnet2gauss', help='model module name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=2048)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=0.7, help='path IoU weight for combined score')
    parser.add_argument('--beta', type=float, default=0.3, help='keypoint IoU weight for combined score')
    parser.add_argument('--save_pred', action='store_true', help='Save prediction for visualization')
    return parser.parse_args()

# -----------------------------
# 主训练函数
# -----------------------------
def main(args):
    # -----------------------------
    # 日志
    # -----------------------------
    model_name = args.model + '_' + str(args.dim) + 'd'
    experiment_dir = os.path.join('results/model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(msg):
        logger.info(msg)
        print(msg)

    # 随机种子
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    else:
        print("Random seed not set")

    # -----------------------------
    # 数据集
    # -----------------------------
    env_type = 'random_' + str(args.dim) + 'd'
    log_string(f"env_type: {env_type}")

    train_dataset = PathPlanDataset(dataset_filepath=f'data/{env_type}/train.npz')
    val_dataset = PathPlanDataset(dataset_filepath=f'data/{env_type}/val.npz')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    log_string(f"The number of training samples: {len(train_dataset)}")
    log_string(f"The number of validation samples: {len(val_dataset)}")

    # -----------------------------
    # 模型和损失
    # -----------------------------
    MODEL = importlib.import_module('model.' + args.model)
    model = MODEL.get_model(coord_dim=args.dim, feature_dim=3).cuda()
    criterion = MODEL.get_loss().cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # -----------------------------
    # 最优模型初始化
    # -----------------------------
    best_score = 0.0
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} - Train"):
            pc_xyz_raw, pc_xyz, pc_features, labels, token = batch

            points = torch.cat([pc_xyz, pc_features], dim=-1).float().cuda()
            points = points.permute(0, 2, 1)
            labels = labels.float().cuda()

            optimizer.zero_grad()
            path_pred, keypoint_pred = model(points)
            pred = torch.stack([path_pred, keypoint_pred], dim=-1)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * points.size(0)
            total_train_samples += points.size(0)

        train_loss /= total_train_samples
        log_string(f"Epoch {epoch+1}/{args.epoch} - Train Loss: {train_loss:.6f}")

        # -----------------------------
        # 验证集
        # -----------------------------
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        intersection_path = 0
        union_path = 0
        intersection_keypoint = 0
        union_keypoint = 0
        total_correct_path = 0
        total_correct_keypoint = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epoch} - Val"):
                pc_xyz_raw, pc_xyz, pc_features, labels, token = batch

                points = torch.cat([pc_xyz, pc_features], dim=-1).float().cuda()
                points = points.permute(0, 2, 1)
                labels = labels.float().cuda()

                path_pred, keypoint_pred = model(points)
                pred = torch.stack([path_pred, keypoint_pred], dim=-1)
                loss = criterion(pred, labels)

                val_loss += loss.item() * points.size(0)
                total_val_samples += points.size(0)

                # 二值化
                path_pred_bin = (path_pred > 0.5).cpu().numpy()
                keypoint_pred_bin = (keypoint_pred > 0.5).cpu().numpy()
                path_label_bin = (labels[..., 0].cpu().numpy() > 0.5)
                keypoint_label_bin = (labels[..., 1].cpu().numpy() > 0.5)

                intersection_path += np.sum(path_pred_bin & path_label_bin)
                union_path += np.sum(path_pred_bin | path_label_bin)
                intersection_keypoint += np.sum(keypoint_pred_bin & keypoint_label_bin)
                union_keypoint += np.sum(keypoint_pred_bin | keypoint_label_bin)

                total_correct_path += np.sum(path_pred_bin == path_label_bin)
                total_correct_keypoint += np.sum(keypoint_pred_bin == keypoint_label_bin)

                # 保存预测（可选）
                if args.save_pred:
                    save_dir = os.path.join(experiment_dir, 'predictions')
                    os.makedirs(save_dir, exist_ok=True)
                    np.savez(os.path.join(save_dir, f"{token[0]}_epoch{epoch+1}.npz"),
                             path_pred=path_pred.cpu().numpy(),
                             keypoint_pred=keypoint_pred.cpu().numpy(),
                             path_label=path_label_bin,
                             keypoint_label=keypoint_label_bin)

        val_loss /= total_val_samples
        path_iou = intersection_path / (union_path + 1e-6)
        keypoint_iou = intersection_keypoint / (union_keypoint + 1e-6)
        path_acc = total_correct_path / total_val_samples
        keypoint_acc = total_correct_keypoint / total_val_samples

        log_string(f"Val Loss: {val_loss:.6f}, Path IoU: {path_iou:.4f}, Keypoint IoU: {keypoint_iou:.4f}")
        log_string(f"Path Acc: {path_acc:.4f}, Keypoint Acc: {keypoint_acc:.4f}")

        # -----------------------------
        # 保存综合最优模型
        # -----------------------------
        score = args.alpha * path_iou + args.beta * keypoint_iou
        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score
            }, best_model_path)
            log_string(f"Saved BEST model at epoch {epoch+1}, Score={best_score:.4f}, Path IoU={path_iou:.4f}, Keypoint IoU={keypoint_iou:.4f}")

# -----------------------------
# 入口
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
