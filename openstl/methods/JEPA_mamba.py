import torch
from openstl.models import JEPA_Mamba_Model
from openstl.utils import (reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method


class MNIST_JEPA_Mamba(Base_method):

    def __init__(self, **args):
        super().__init__(**args)
        self.constraints = self._get_constraints()

    def _build_model(self, **args):
        print("args: ", args)
        print("hparams: ", self.hparams)
        return JEPA_Mamba_Model(configs=args)

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def forward(self, batch_x, batch_y, **kwargs):
        # print("batch_x: ", batch_x.shape)
        # print("batch_y: ", batch_y.shape)
        # Concatenate the input and output tensors
        ims = torch.cat([batch_x, batch_y], dim=1).contiguous()
        # If there is a decoder, it should be called here
        # if self.hparams.train_decoder:
        pred_y, _ = self.model(ims, latent_tensor=None)
        # else:
        #     pred_y = batch_y
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        # print("batch_x: ", batch_x.shape)
        # print("batch_y: ", batch_y.shape)
        # Concatenate the input and output tensors
        ims = torch.cat([batch_x, batch_y], dim=1).contiguous()
        if self.hparams.latent_tensor_mode == 2:
            pred_y, loss = self.model(ims, None)
        else:
            # Extract the latent tensor from the batch tensor
            latent_tensor = batch_x[:, 0]
            pred_y, loss = self.model(ims, latent_tensor)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    