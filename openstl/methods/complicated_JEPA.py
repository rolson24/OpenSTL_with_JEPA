import torch
from openstl.models import Complicated_JEPA_Model
from openstl.utils import (reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method


class Bounching_Shapes_Complicated_JEPA(Base_method):

    def __init__(self, **args):
        super().__init__(**args)
        self.constraints = self._get_constraints()

    def _build_model(self, **args):
        print("args: ", args)
        print("hparams: ", self.hparams)
        configs = args['configs']
        return Complicated_JEPA_Model(configs=configs, **args)

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
        pred_y, loss = self.model(ims)

        # print(f"loss: {loss}")

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        # print("batch_x: ", batch_x.shape)
        # print("batch_y: ", batch_y.shape)
        # Concatenate the input and output tensors
        ims = torch.cat([batch_x, batch_y], dim=1).contiguous()
        pred_y, loss = self.model(ims)

        # print(f"loss: {loss}")

        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss