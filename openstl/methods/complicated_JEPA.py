import torch
from openstl.models import Complicated_JEPA_Model
from openstl.utils import (reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method


class Bounching_Shapes_Complicated_JEPA(Base_method):

    def __init__(self, **args):
        super().__init__(**args)
        self.constraints = self._get_constraints()
        if 'train_linear_probe' in args:
            self.train_linear_probe = args['train_linear_probe']
        else:
            self.train_linear_probe = False

    def _build_model(self, **args):
        print("args: ", args)
        print("hparams: ", self.hparams)
        if 'train_linear_probe' in args:
            self.train_linear_probe = args['train_linear_probe']
            model = Complicated_JEPA_Model(**args)
            model.freeze_encoder()
        else:
            self.train_linear_probe = False
            model = Complicated_JEPA_Model(**args)
        # configs = args['configs']
        return model

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(0, 7):
            for j in range(0, 7):
                constraints[ind,i,j] = 1
                ind +=1
        return constraints 

    def load_state_dict(self, checkpoint, strict=False):
        """Load state dict with support for linear probe training"""
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Check if keys have 'model.' prefix and remove it
        if any(k.startswith('model.') for k in list(state_dict.keys())[:5]):  # Check first few keys
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            print("Removed 'model.' prefix from state dict keys")
            
        if self.train_linear_probe:
            # Remove linear probe related weights if they exist
            # filtered_state_dict = {k: v for k, v in state_dict.items() if 'linear_head' not in k}
            self.model.load_state_dict(state_dict, strict=False)
            print("Loaded pre-trained model weights without linear probe layers")
        else:
            # Load full model weights
            self.model.load_state_dict(state_dict, strict=strict)
            print("Loaded full model weights")
    
    def load_pretrained_model(self, checkpoint_path, reset_linear_probes=True):
        """Load a pre-trained model while optionally resetting linear probe weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if reset_linear_probes and isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict'] 
            else:
                state_dict = checkpoint
                
            # Remove linear probe weights to initialize them randomly
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'linear_probe' not in k}
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded pre-trained model with fresh linear probe layers")
        else:
            # Load the complete state dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print("Loaded complete pre-trained model")
        
        # If training linear probes, freeze the encoder and predictor
        if self.train_linear_probe:
            self._freeze_backbone_parameters()
    
    def _freeze_backbone_parameters(self):
        """Freeze all non-linear probe parameters when training linear probes"""
        for name, param in self.model.named_parameters():
            if 'linear_probe' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Froze backbone parameters, only linear probe weights will be updated")

    def forward(self, batch_x, batch_y, **kwargs):
        ims = torch.cat([batch_x, batch_y], dim=1).contiguous()
        pred_y, _ = self.model(ims, latent_tensor=None)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y, labels = batch

        if self.train_linear_probe:
            pred_labels, loss = self.model.forward_linear_probe(batch_x, labels)
            self.log('train_probe_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        else:
            # Concatenate the input and output tensors
            ims = torch.cat([batch_x, batch_y], dim=1).contiguous()
            pred_y, loss = self.model(ims)
            self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, labels = batch
        if self.train_linear_probe:
            pred_labels, loss = self.model.forward_linear_probe(batch_x, labels)
            self.log('val_probe_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            # Also log as val_loss for the ModelCheckpoint callback
            self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=False)
        else:
            # Concatenate the input and output tensors
            ims = torch.cat([batch_x, batch_y], dim=1).contiguous()
            pred_y, loss = self.model(ims)
            self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss