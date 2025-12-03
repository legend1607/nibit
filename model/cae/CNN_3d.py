import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

import torch.nn as nn

class Encoder_CNN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 2, 3, 1, 1),
            nn.BatchNorm3d(2), 
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(2, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(4, 8, 3, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        # final shape = (B, 32, 1, 1, 1)
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 1 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 60)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.fc_layers(x)
        return x

