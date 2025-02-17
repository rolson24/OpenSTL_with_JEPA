import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the encoder module for a Joint Embedding Prediction Architecture (JEPA) model
# Description:
# In experiments with MovingMNIST, we model the encoder as a 5-layer convolutional neural network with
# batch norm and ReLU activation function at each layer. It has 3 spatial and 2 temporal convolutions followed
# by an average pooling layer. The predictor is an MLP with 2 hidden layers. Unless otherwise noted, the input
# x contains 3 frames and the target y contains the following 12 frames in the video. The predictor outputs the
# hidden state of the 12 target frames y simultaneously.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingMNISTEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(MovingMNISTEncoder, self).__init__()

        self.in_channels = in_channels
        self.spatial_output_channels = 256
        
        # 3 spatial convolutions
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=self.spatial_output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 2 temporal convolutions
        self.conv4 = nn.Conv3d(in_channels=self.spatial_output_channels, out_channels=512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        
        self.conv5 = nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(1024)
        
        # Average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        batch, length, channel, height, width = x.shape
        
        # Apply spatial convolutions to each frame
        x = x.view(batch * length, channel, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Reshape for temporal convolutions
        x = x.view(batch, length, self.spatial_output_channels, height, width)
        
        # Apply temporal convolutions
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Apply average pooling
        x = self.avg_pool(x)
        
        return x.squeeze()  # Remove extra dimensions

# Example usage
# encoder = MovingMNISTEncoder()
# input_tensor = torch.randn(1, 3, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 resolution
# output = encoder(input_tensor)
# print(output.shape)