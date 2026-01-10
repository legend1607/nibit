
import os
import logging
import argparse
import importlib

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from model.PNG.PathPlanDataLoader import PathPlanDataset


classes = ['other free points', 'optimal path points']
NUM_CLASSES = len(classes)
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=2000, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--random_seed', type=int, default=None)
    return parser.parse_args()


class get_loss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super(get_loss, self).__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, pred, target, trans_feat, weight):
        """
        pred: (B*N, C)  log-probabilities (because model uses log_softmax)
        target: (B*N,)
        weight: (C,) class weights (your TRAIN_DATASET.labelweights)
        """
        # logpt: (B*N,)
        logpt = pred.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()  # pt in (0,1)

        # focal factor
        focal = (1.0 - pt).pow(self.gamma)

        # alpha / class weight per-sample
        if weight is not None:
            alpha_t = weight.gather(0, target)  # (B*N,)
            loss = -alpha_t * focal * logpt
        else:
            loss = -focal * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    model_name = args.model
    experiment_dir = os.path.join('results/model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.random_seed is not None:
        print("Setting random seed to {0}".format(args.random_seed))
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    else:
        print("Random seed not set")
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    log_string("saving to "+experiment_dir)
    
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    env_type = 'liche'
    print("env_type: ", env_type)

    TRAIN_DATASET = PathPlanDataset(dataset_filepath='data/'+env_type+'/train.npz')
    VAL_DATASET = PathPlanDataset(dataset_filepath='data/'+env_type+'/val.npz')
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=10,
        drop_last=False,
    )
    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=10,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = torch.as_tensor(TRAIN_DATASET.labelweights, dtype=torch.float32, device=device)
    weights = weights / weights.mean()

    print("LOSS WEIGHTS:", weights.detach().cpu().numpy())


    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validation data is: %d" % len(VAL_DATASET))

    MODEL = importlib.import_module('model.PNG.'+args.model)
    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    criterion = get_loss(gamma=args.focal_gamma).to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_'+model_name+'.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_optimal_path_IoU = None

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, batch in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            pc_xyz_raw, pc_xyz, pc_features, pc_labels, token = batch
            assert pc_xyz.shape[-1] == 4, pc_xyz.shape
            assert pc_features.shape[-1] == 3, pc_features.shape
            assert pc_xyz.shape[1] == NUM_POINT, pc_xyz.shape


            pc_xyz = pc_xyz.float() 
            points = torch.cat([pc_xyz, pc_features], dim=2) # (b, N, 7)
            points, target = points.float().to(device), pc_labels.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            target = target.view(-1)
            batch_label = target.detach().cpu().numpy()

            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            bs = pc_xyz.shape[0]
            total_seen += (bs * NUM_POINT)
            loss_sum += loss.item()

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        with torch.no_grad():
            num_batches = len(valDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, batch in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                pc_xyz_raw, pc_xyz, pc_features, pc_labels, token = batch

                assert pc_xyz.shape[-1] == 4, pc_xyz.shape
                assert pc_features.shape[-1] == 3, pc_features.shape
                assert pc_xyz.shape[1] == NUM_POINT, pc_xyz.shape

                pc_xyz = pc_xyz.float()
                points = torch.cat([pc_xyz, pc_features], dim=2)  # (B, N, 7)
                points, target = points.float().to(device), pc_labels.long().to(device)

                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)  # seg_pred: (B, N, C)

                # ✅ 改这里：直接 torch argmax
                pred_val = seg_pred.argmax(dim=2).cpu().numpy()  # (B, N)

                # loss 仍然用 (B*N, C) 形式
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().numpy()  # (B, N)
                target = target.view(-1)           # (B*N,)
                loss = criterion(seg_pred, target, trans_feat, weights)

                correct = np.sum(pred_val == batch_label)
                total_correct += correct
                bs = pc_xyz.shape[0]
                total_seen += (bs * NUM_POINT)
                loss_sum += loss.item()

                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            optimal_path_IoU = total_correct_class[1] / float(total_iou_deno_class[1])
            if best_optimal_path_IoU is None or optimal_path_IoU >= best_optimal_path_IoU:
                best_optimal_path_IoU = optimal_path_IoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_'+model_name+'.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best Optimal Path IoU: %f' % best_optimal_path_IoU)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
