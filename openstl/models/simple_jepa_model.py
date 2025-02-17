import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import MovingMNISTJEPAEncoder, MovingMNISTJEPAPredictor

class SimpleJEPA_Model(nn.Module):
    def __init__(self):
        super(SimpleJEPA_Model, self).__init__()
        self.encoder = MovingMNISTJEPAEncoder()

        self.predictor = MovingMNISTJEPAPredictor()

    def forward(self, frames_tensor, mask_true, **kwargs):
        x = self.encoder(x)
        x = self.predictor(x)
        # Calculate loss
        return x
