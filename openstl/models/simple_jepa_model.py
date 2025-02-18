import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import MovingMNISTJEPAEncoder, MovingMNISTJEPAPredictor, MovingMNSITJEPADecoder

class SimpleJEPA_Model(nn.Module):
    r"""SimpleJEPA Model

    Implementation of `Video Representation Learning with Joint-Embedding Predictive Architectures <https://arxiv.org/pdf/2412.10925>`_.

    """
    def __init__(self, configs, **kwargs):
        super(SimpleJEPA_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs

        self.input_encoder = MovingMNISTJEPAEncoder(in_channels=C, out_channels=configs.embed_dim)

        self.predictor = MovingMNISTJEPAPredictor(in_channels=configs.embed_dim, out_channels=configs.embed_dim)

        # Now create the target encoder that has a momentum average of the weights of the input encoder
        self.target_encoder = MovingMNISTJEPAEncoder(in_channels=C, out_channels=configs.embed_dim)
        # Copy the weights of the input encoder to the target encoder
        self.target_encoder.load_state_dict(self.input_encoder.state_dict())
        # Make the target encoder not trainable
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        # How to update the target encoder
        self.target_encoder_update_rate = 0.999


        # Now create a decoder that will take the output of the predictor to generate the next frames
        self.decoder = MovingMNSITJEPADecoder(in_channels=configs.embed_dim, image_size=H)


    def forward(self, frames_tensor, latent_tensor, mask_true, **kwargs):
        # Get first 3 frames from the input'
        x = frames_tensor[:, :3]
        # Encode the frames
        hx = self.input_encoder(x)
        hx = self.predictor(hx, latent_tensor)

        y = frames_tensor[:, 3:]
        hy = self.target_encoder(y)

        y_hat = self.decoder(hy)

        # Calculate all the losses

        return x
