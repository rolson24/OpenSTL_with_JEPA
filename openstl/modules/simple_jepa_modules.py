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

# In the first setting, z is modeled as a discrete latent variable. In particular, it is a one-hot vector of dimension
# 5 (which is the number of possible random switches in the trajectory by design). The discrete latent z
# influences the top linear layer of the Predictor, which can be one of 5 options based on the value of z. In
# other words, the active component in z selects the last linear layer of the Predictor.
class MovingMNISTJEPAPredictor(nn.Module):
    def __init__(self, in_channels=1024, in_frames=3, out_channels=1024, out_frames=12, hidden_size=2048, latent_vector_mode=1, latent_vector_size=5):
        super(MovingMNISTJEPAPredictor, self).__init__()

        self.latent_vector_mode = latent_vector_mode
        self.latent_vector_size = latent_vector_size

        self.in_frames = in_frames
        self.out_frames = out_frames

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_size = in_channels * in_frames
        self.out_size = out_channels * out_frames

        self.hidden_size = hidden_size
        
        # 2 hidden layers
        self.fc1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        if latent_vector_mode == 1:
            # 5 output layers that get selected based on the latent vector
            self.fc3_options = nn.ModuleList([
                nn.Linear(self.hidden_size, self.out_size) for _ in range(latent_vector_size)
            ])
        elif latent_vector_mode == 2:
            # In mode 2 we have stored a latent vector in the input and every optimization step, we update the latent vector with the gradient of the loss
            # 1 output layer with latent vector concatenated to the input
            self.fc3 = nn.Linear(self.hidden_size + latent_vector_size, self.out_size)
        


        
    def forward(self, hx, z):
        """
        hx: Tensor of shape (B, T_in, d)
           T_in is the number of input frames (e.g., 3).
           d is the feature dimension per frame.
        z: Tensor of shape (B, latent_dim) -- one-hot vectors of dimension 5.
        output: Tensor of shape (B, output_dim)
           where output_dim = (T_out * d), T_out being the number of target frames (e.g., 12).
        """

        B, T_in, d = hx.shape
        # Flatten hx by concatenating the features of the input frames.
        hx_flat = hx.view(B, T_in * d)  # shape: (B, T_in*d)
        
        h = F.relu(self.fc1(hx_flat))
        h = F.relu(self.fc2(h))

        if self.latent_vector_mode == 1:
        
            # Compute outputs for each latent option
            # Each option yields a tensor of shape (B, output_dim)
            outputs = torch.stack([fc3(h) for fc3 in self.fc3_options], dim=1)  # shape: (B, latent_dim, output_dim)
            
            # Combine the outputs using the one-hot latent vector z.
            # For a one-hot vector, this will effectively select the corresponding output.
            z = z.unsqueeze(-1)  # shape: (B, latent_dim, 1)
            output = torch.sum(outputs * z, dim=1)  # shape: (B, output_dim)
        elif self.latent_vector_mode == 2:
            # Concatenate the latent vector to the hidden state
            h = torch.cat([h, z], dim=1)
            output = self.fc3(h)  # shape: (B, output_dim)

        # Reshape to (B, T_out, d) where d is the output channels
        output = output.view(B, self.out_frames, self.out_channels)
        
        return output


# Create a decoder module that can decode the output of the predictor
class MovingMNISTJEPADecoder(nn.Module):
    def __init__(self, out_channels=3, image_size=64):
        super(MovingMNISTJEPADecoder, self).__init__()
        self.out_channels = out_channels
        self.image_size = image_size
        
        # Reverse of temporal conv (1D) layers
        self.convT4 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.convT5 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Reverse of spatial conv (2D) layers
        self.convT1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.convT3 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (B, T, 1024)
        B, T, _ = x.shape
        
        # Go back through temporal path
        x = x.transpose(1, 2)  # (B, 1024, T)
        x = F.relu(self.bn4(self.convT4(x)))  # (B, 512, T)
        x = F.relu(self.bn5(self.convT5(x)))  # (B, 256, T)
        x = x.transpose(1, 2)  # (B, T, 256)
        
        # Flatten (B, T) => (B*T, ...)
        x = x.view(B * T, 256, 1, 1)  # (B*T, 256, 1, 1)

        # Adjust if you upsample differently (e.g., via stride or interpolate)
        # Here we assume final 64x64 output frames:
        x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
    
        # Go back through spatial path
        x = F.relu(self.bn1(self.convT1(x)))  # (B*T, 128, H, W)
        x = F.relu(self.bn2(self.convT2(x)))  # (B*T, 64, H, W)
        x = self.convT3(x)                    # (B*T, out_channels, H, W)

        # Reshape to (B, T, C, H, W)
        x = x.view(B, T, self.out_channels, self.image_size, self.image_size)
        
        return x






if __name__ == "__main__":
    encoder = MovingMNISTJEPAEncoder()
    input_tensor = torch.randn(1, 3, 3, 64, 64)  # Batch size 1, 3 frames, 3 channels, 64x64 resolution
    output = encoder(input_tensor)
    print(output.shape)

    predictor = MovingMNISTJEPAPredictor(latent_vector_mode=1, latent_vector_size=5)
    input_tensor = torch.randn(1, 3, 1024)  # Batch size 1, 3 frames, 1024 channels
    z = torch.randn(1, 5)  # Batch size 1, latent vector size 5
    output = predictor(input_tensor, z)
    print(output.shape)

    # Test the decoder
    decoder = MovingMNISTJEPADecoder()
    input_tensor = torch.randn(1, 12, 1024)  # Batch size 1, 12 frames, 1024 channels
    output = decoder(input_tensor)
    print(output.shape)
