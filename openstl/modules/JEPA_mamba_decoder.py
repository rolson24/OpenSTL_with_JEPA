import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAImageDecoder(nn.Module):
    """
    Simple MLP-based decoder for transforming JEPA embeddings back to image space.
    """
    def __init__(self, in_channels=1024, hidden_size=2048, img_channels=1, img_height=64, img_width=64):
        super(JEPAImageDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        
        # Calculate output size
        self.img_size = img_channels * img_height * img_width
        
        # MLP layers to transform embeddings to image space
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.img_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        x: Tensor of shape (B, T, d) where:
           B is batch size
           T is number of frames
           d is embedding dimension per frame
        
        Returns: Tensor of shape (B, T, C, H, W) with reconstructed frames
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Reshape to process all frames at once
        x = x.reshape(-1, embed_dim)  # (B*T, d)
        
        # Pass through MLP
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to image dimensions
        x = x.view(batch_size, seq_len, self.img_channels, self.img_height, self.img_width)
        
        return x


class JEPAConvImageDecoder(nn.Module):
    """
    A slightly more sophisticated decoder using both MLP and transposed convolutions.
    More parameter efficient while preserving spatial relationships.
    """
    def __init__(self, in_channels=1024, hidden_size=1024, img_channels=1, img_height=64, img_width=64):
        super(JEPAConvImageDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        
        # Calculate feature map dimensions (assuming we'll upsample to 8x8 first)
        self.feature_dim = 8
        feature_size = hidden_size
        
        # Initial MLP to get reasonable feature map size
        self.fc = nn.Linear(in_channels, feature_size * self.feature_dim * self.feature_dim)
        
        # Transposed convolution layers for upsampling
        self.deconv1 = nn.ConvTranspose2d(feature_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        x: Tensor of shape (B, T, d) where:
           B is batch size
           T is number of frames
           d is embedding dimension per frame
        
        Returns: Tensor of shape (B, T, C, H, W) with reconstructed frames
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Process all frames at once
        x = x.reshape(-1, embed_dim)  # (B*T, d)
        
        # Initial MLP
        x = self.fc(x)
        
        # Reshape to initial feature map shape
        x = x.view(-1, x.size(1) // (self.feature_dim * self.feature_dim), 
                  self.feature_dim, self.feature_dim)
        
        # Apply transposed convolutions with upsampling
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)  # Final layer without ReLU for full output range
        
        # Reshape to add back the time dimension
        x = x.reshape(batch_size, seq_len, self.img_channels, self.img_height, self.img_width)
        
        return x
