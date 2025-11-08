import os
import logging
import argparse
import importlib

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from model.pointnet2_utils import pc_normalize

# ------------------------------
# Dataset
# ------------------------------
class PathPlanDataset(Dataset):
    def __init__(self, dataset_filepath):
        data = np.load(dataset_filepath)
        self.pc = data['pc'].astype(np.float32)
        self.start_mask = data['start'].astype(np.float32)
        self.goal_mask = data['goal'].astype(np.float32)
        self.free_mask = data['free'].astype(np.float32)
        self.path_mask = data["path"].astype(np.float32)
        self.keypoint_mask = data["keypoint"].astype(np.float32)
        self.token = data['token']
        self.d = self.pc.shape[2]
        self.n_points = self.pc.shape[1]
        print(f"[PathPlanDataset] Loaded point cloud with dimension = {self.d}, n_points = {self.n_points}")

    def __len__(self):
        return len(self.pc)
    
    def __getitem__(self, index):
        pc_xyz_raw = self.pc[index]
        pc_xyz = pc_normalize(pc_xyz_raw)
        pc_features = np.stack(
            (self.start_mask[index], self.goal_mask[index], self.free_mask[index]),
            axis=-1,
        )
        pc_labels = self.path_mask[index]
        keypoint_labels = self.keypoint_mask[index]
        return pc_xyz_raw, pc_xyz, pc_features, pc_labels, keypoint_labels, self.token[index]

# ------------------------------
# ReLU In-place
# ------------------------------
def inplace_relu(m):
    if isinstance(m, torch.nn.ReLU):
        m.inplace = True

# ------------------------------
# Arguments
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--env', type=str, default='random_2d', choices=['random', 'kuka_random'])
    parser.add_argument('--dim', type=int, default=2, help='environment dimension: 2 or 3.')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=2048)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--random_seed', type=int, default=None)
    return parser.parse_args()

# ------------------------------
# Weighted Soft BCE for soft labels
# ------------------------------
class WeightedSoftBCE(nn.Module):
    def __init__(self, path_weight=1.0, keypoint_weight=1.0, pos_weight=5.0):
        super(WeightedSoftBCE, self).__init__()
        self.path_weight = path_weight
        self.keypoint_weight = keypoint_weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())

    def forward(self, pred_path, pred_keypoint, target_path, target_keypoint):
        target_path = target_path.float()
        target_keypoint = target_keypoint.float()
        loss_path = self.criterion(pred_path, target_path)
        loss_keypoint = self.criterion(pred_keypoint, target_keypoint)
        total_loss = self.path_weight * loss_path + self.keypoint_weight * loss_keypoint
        return total_loss, loss_path, loss_keypoint

# ------------------------------
# Soft IoU/F1 metrics
# ------------------------------
def compute_metrics_soft(pred_path, target_path, pred_keypoint, target_keypoint):
    # Soft IoU for path
    intersection = np.sum(pred_path * target_path)
    union = np.sum(pred_path + target_path - pred_path * target_path)
    iou_path = intersection / (union + 1e-6)

    # Soft F1 for keypoints
    tp = np.sum(pred_keypoint * target_keypoint)
    fp = np.sum(pred_keypoint * (1 - target_keypoint))
    fn = np.sum((1 - pred_keypoint) * target_keypoint)
    f1_key = 2 * tp / (2*tp + fp + fn + 1e-6)

    return iou_path, f1_key

# ------------------------------
# Training
# ------------------------------
def main(args):
    model_name = args.model + f"_{args.dim}d"
    experiment_dir = os.path.join('results/model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, model_name + '.txt'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(s):
        logger.info(s)
        print(s)

    log_string(args)
    log_string("Saving to "+experiment_dir)

    # ------------------------------
    # Datasets
    # ------------------------------
    TRAIN_DATASET = PathPlanDataset(dataset_filepath=f'data/{args.env}/train.npz')
    VAL_DATASET = PathPlanDataset(dataset_filepath=f'data/{args.env}/val.npz')

    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False
    )

    # ------------------------------
    # Model
    # ------------------------------
    MODEL = importlib.import_module('model.' + args.model)
    classifier = MODEL.get_model(coord_dim=TRAIN_DATASET.d).cuda()
    criterion = WeightedSoftBCE(path_weight=1.0, keypoint_weight=1.0).cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    start_epoch = 0
    best_score = None
    try:
        checkpoint = torch.load(os.path.join(checkpoints_dir, f'best_{model_name}.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        log_string("Loaded pre-trained model")
    except:
        classifier.apply(weights_init)
        log_string("Training from scratch")

    # ------------------------------
    # Optimizer
    # ------------------------------
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    # ------------------------------
    # TensorBoard
    # ------------------------------
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    # ------------------------------
    # Training loop
    # ------------------------------
    for epoch in range(start_epoch, args.epoch):
        classifier.train()
        loss_sum = 0
        total_seen = 0
        total_iou_path = 0
        total_f1_key = 0

        # Learning rate decay
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), 1e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        log_string(f"**** Epoch {epoch+1} **** Learning rate: {lr}")

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            pc_xyz_raw, pc_xyz, pc_features, pc_labels, keypoint_labels, _ = batch
            pc_xyz = torch.Tensor(pc_xyz).float()
            points = torch.cat([pc_xyz, pc_features], dim=2).float().cuda()
            points = points.transpose(2,1)

            target_path = pc_labels.view(-1,1).float().cuda()
            target_keypoint = keypoint_labels.view(-1,1).float().cuda()

            # Forward + Loss
            path_logits, keypoint_logits = classifier(points)
            path_logits_flat = path_logits.transpose(2,1).contiguous().view(-1,1)
            keypoint_logits_flat = keypoint_logits.transpose(2,1).contiguous().view(-1,1)

            loss, loss_path, loss_keypoint = criterion(
                path_logits_flat, keypoint_logits_flat, target_path, target_keypoint
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            # Metrics (soft IoU/F1)
            pred_path_np = torch.sigmoid(path_logits_flat).detach().cpu().numpy()
            pred_key_np = torch.sigmoid(keypoint_logits_flat).detach().cpu().numpy()
            iou_path, f1_key = compute_metrics_soft(pred_path_np, target_path.cpu().numpy(),
                                                    pred_key_np, target_keypoint.cpu().numpy())
            total_iou_path += iou_path * target_path.shape[0]
            total_f1_key += f1_key * target_keypoint.shape[0]
            total_seen += target_path.shape[0]

        # Epoch metrics
        epoch_loss = loss_sum / len(train_loader)
        epoch_iou_path = total_iou_path / total_seen
        epoch_f1_key = total_f1_key / total_seen

        log_string(f"Train Loss: {epoch_loss:.4f}, IoU Path: {epoch_iou_path:.4f}, F1 Keypoint: {epoch_f1_key:.4f}")
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/IoU_Path', epoch_iou_path, epoch)
        writer.add_scalar('Train/F1_Keypoint', epoch_f1_key, epoch)

        # ------------------------------
        # Validation
        # ------------------------------
        classifier.eval()
        val_loss_sum = 0
        val_total_seen = 0
        val_total_iou_path = 0
        val_total_f1_key = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                pc_xyz_raw, pc_xyz, pc_features, pc_labels, keypoint_labels, _ = batch
                pc_xyz = torch.Tensor(pc_xyz).float()
                points = torch.cat([pc_xyz, pc_features], dim=2).float().cuda()
                points = points.transpose(2,1)

                target_path = pc_labels.view(-1,1).float().cuda()
                target_keypoint = keypoint_labels.view(-1,1).float().cuda()

                path_logits, keypoint_logits = classifier(points)
                path_logits_flat = path_logits.transpose(2,1).contiguous().view(-1,1)
                keypoint_logits_flat = keypoint_logits.transpose(2,1).contiguous().view(-1,1)

                loss, _, _ = criterion(path_logits_flat, keypoint_logits_flat, target_path, target_keypoint)
                val_loss_sum += loss.item()

                # Metrics
                pred_path_np = torch.sigmoid(path_logits_flat).detach().cpu().numpy()
                pred_key_np = torch.sigmoid(keypoint_logits_flat).detach().cpu().numpy()
                iou_path, f1_key = compute_metrics_soft(pred_path_np, target_path.cpu().numpy(),
                                                        pred_key_np, target_keypoint.cpu().numpy())
                val_total_iou_path += iou_path * target_path.shape[0]
                val_total_f1_key += f1_key * target_keypoint.shape[0]
                val_total_seen += target_path.shape[0]

        val_loss = val_loss_sum / len(val_loader)
        val_iou_path = val_total_iou_path / val_total_seen
        val_f1_key = val_total_f1_key / val_total_seen

        log_string(f"Val Loss: {val_loss:.4f}, IoU Path: {val_iou_path:.4f}, F1 Keypoint: {val_f1_key:.4f}")
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/IoU_Path', val_iou_path, epoch)
        writer.add_scalar('Val/F1_Keypoint', val_f1_key, epoch)

        # ------------------------------
        # Save best model (weighted score)
        # ------------------------------
        score = 0.6 * val_iou_path + 0.4 * val_f1_key
        if best_score is None or score >= best_score:
            best_score = score
            savepath = os.path.join(checkpoints_dir, f'best_{model_name}.pth')
            log_string(f"Saving model at {savepath} with weighted score: {score:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, savepath)

    writer.close()

# ------------------------------
# Entry
# ------------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)
