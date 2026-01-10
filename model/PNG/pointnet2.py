import torch.nn as nn
import torch.nn.functional as F

from model.PNG.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.06, 0.12], [16, 32], 7, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.12, 0.24], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.24, 0.48], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.48, 0.96], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, joint):
        l0_points = joint
        l0_joint = joint[:,:4,:]

        l1_joint, l1_points = self.sa1(l0_joint, l0_points)
        l2_joint, l2_points = self.sa2(l1_joint, l1_points)
        l3_joint, l3_points = self.sa3(l2_joint, l2_points)
        l4_joint, l4_points = self.sa4(l3_joint, l3_points)

        l3_points = self.fp4(l3_joint, l4_joint, l3_points, l4_points)
        l2_points = self.fp3(l2_joint, l3_joint, l2_points, l3_points)
        l1_points = self.fp2(l1_joint, l2_joint, l1_points, l2_points)
        l0_points = self.fp1(l0_joint, l1_joint, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points
