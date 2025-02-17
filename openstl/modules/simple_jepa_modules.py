import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the encoder module for a Joint Embedding Prediction Architecture (JEPA) model
# Description:
# In experiments with MovingMNIST, we model the encoder as a 5-layer convolutional neural network with
# batch norm and ReLU activation function at each layer. It has 3 spatial and 2 temporal convolutions followed
# by an average pooling layer.

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingMNISTJEPAEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(MovingMNISTJEPAEncoder, self).__init__()
        self.in_channels = in_channels
        self.spatial_output_channels = 256
        
        # 3 spatial convolutions
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=self.spatial_output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.spatial_output_channels)
        
        # 2 temporal convolutions (1D along time axis)
        self.conv4 = nn.Conv1d(in_channels=self.spatial_output_channels, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Note: we use adaptive_avg_pool2d to pool over spatial dims.
        # The provided AdaptiveAvgPool3d could be used if you reshape appropriately,
        # but here we keep the per-frame information intact.
        
    def forward(self, x):
        # x shape: (batch, time, channel, height, width)
        B, T, C, H, W = x.shape
        
        # Process each frame separately using 2D convolutions:
        # Combine batch and time dimensions for spatial processing
        x = x.view(B * T, C, H, W)  # shape: (B*T, C, H, W)
        
        # Spatial convolutional layers with ReLU activations
        x = F.relu(self.bn1(self.conv1(x)))  # (B*T, 64, H, W)
        x = F.relu(self.bn2(self.conv2(x)))  # (B*T, 128, H, W)
        x = F.relu(self.bn3(self.conv3(x)))  # (B*T, 256, H, W)
        
        # Global average pooling over spatial dimensions to get a feature vector per frame
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B*T, 256, 1, 1)
        x = x.view(B, T, self.spatial_output_channels)  # (B, T, 256)
        
        # Prepare for temporal convolutions: transpose to shape (B, channels, T)
        x = x.transpose(1, 2)  # (B, 256, T)
        
        # Temporal convolutional layers with ReLU activations
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 512, T)
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 1024, T)
        
        # Permute back to (B, T, d) where d=1024 so that each frame gets its feature vector
        x = x.transpose(1, 2)  # (B, T, 1024)
        
        return x



# Now the prediction module
# Description:
# The predictor is an MLP with 2 hidden layers. Unless otherwise noted, the input
# x contains 3 frames and the target y contains the following 12 frames in the video. The predictor outputs the
# hidden state of the 12 target frames y simultaneously.
class MovingMNISTJEPAPredictor(nn.Module):
    def __init__(self, in_channels=1024, in_frames=3, out_channels=1024, out_frames=12, hidden_size=2048, latent_vector_on=False, latent_vector_size=128):
        super(MovingMNISTJEPAPredictor, self).__init__()

        self.latent_vector_size = latent_vector_size if latent_vector_on else 0

        self.in_frames = in_frames
        self.out_frames = out_frames

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_size = in_channels * in_frames + self.latent_vector_size
        self.out_size = out_channels * out_frames

        self.hidden_size = hidden_size
        
        # 2 hidden layers
        self.fc1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.out_size)

        # Latent vector
        if latent_vector_on:
            self.latent_vector = nn.Parameter(torch.randn(1, self.latent_vector_size))
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.shape[0], -1)
        # Append latent vector if enabled
        if self.latent_vector_size > 0:
            B, TD = x.shape
            x = torch.cat([x, self.latent_vector.repeat(B, 1)], dim=1)
        # Apply hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to (B, T, C) where C is the output channels
        B, C = x.shape
        x = x.view(B, self.out_frames, self.out_channels)
        
        return x

if __name__ == "__main__":
    encoder = MovingMNISTJEPAEncoder()
    input_tensor = torch.randn(1, 3, 3, 64, 64)  # Batch size 1, 3 frames, 3 channels, 64x64 resolution
    output = encoder(input_tensor)
    print(output.shape)

    predictor = MovingMNISTJEPAPredictor(latent_vector_on=True, latent_vector_size=128)
    input_tensor = torch.randn(1, 3, 1024)  # Batch size 1, 3 frames, 1024 channels
    output = predictor(input_tensor)
    print(output.shape)
