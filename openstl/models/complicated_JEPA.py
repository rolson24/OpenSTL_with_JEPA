import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import MovingMNISTJEPAEncoder, MovingMNISTJEPAPredictor, MovingMNISTJEPADecoder
from openstl.utils.ISTA import ISTA
from openstl.modules import Encoder, Decoder, MidMetaNet


def l_vcr(h, alpha, beta):
    """
    Args:
        h: the hidden state tensor of shape (B, T, d)
        alpha: the weight of the variance term
        beta: the weight of the covariance term
    """
    # print("h: ", h)
    B, T, d = h.shape

    # First we compute the varaiance term
    var = torch.var(h, dim=0, unbiased=False) # shape: (T, d)
    # print("var: ", var)

    # Compute the std deviation
    std = torch.sqrt(var + 1e-6)

    # Apply the hinge loss
    var_term = torch.clamp(1 - std, min=0).mean()

    # Compute the covariance term (vectorized)
    mean_t = h.mean(dim=0, keepdim=True) # shape: (1, T, d)
    h_centered = h - mean_t
    h_centered = h_centered.permute(1, 0, 2) # shape: (T, B, d)
    # print("h_centered:", h_centered)
    # Compute the covariance matrix for each frame
    cov = torch.bmm(h_centered.transpose(1, 2), h_centered) / (B) # shape: (T, d, d)
    # print("cov:", cov)
    # Zero out the diagonal
    mask = 1 - torch.eye(d, device=h.device).unsqueeze(0) # shape: (1, d, d)
    cov_term = ((cov * mask) ** 2).sum() / (T * d)

    # print("cov_term:", cov_term)

    return alpha*var_term + beta*cov_term

class SimpleJEPA_Model(nn.Module):
    r"""SimpleJEPA Model

    Implementation of `Video Representation Learning with Joint-Embedding Predictive Architectures <https://arxiv.org/pdf/2412.10925>`_.

    """
    def __init__(self, configs, **kwargs):
        super(SimpleJEPA_Model, self).__init__()
        T, C, H, W = configs['in_shape']


        # SimVP encoder settings
        hid_S = configs['hid_s'] # hid_S=16
        hid_T = configs['hid_t'] # hid_T=256
        N_S = configs['N_S'] # N_S=4
        N_T = configs['N_T'] # N_T=4
        model_type = configs['model_type'] # model_type='gSTA'
        mlp_ratio = configs['mlp_ratio'] # mlp_ratio=8.
        drop = configs['drop'] # drop=0.0
        drop_path = configs['drop_path'] # drop_path=0.0
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        spatio_kernel_dec=3,

        self.configs = configs

        self.latent_tensor_mode = configs['latent_tensor_mode']
        self.latent_tensor_size = configs['latent_tensor_size']

        self.in_frames = configs['pre_seq_length']
        self.out_frames = configs['aft_seq_length']

        # self.input_encoder = MovingMNISTJEPAEncoder(in_channels=C, out_channels=configs['embed_dim'])

        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)

        # self.predictor = MovingMNISTJEPAPredictor(in_channels=configs['embed_dim'], out_channels=configs['embed_dim'], in_frames=self.in_frames, out_frames=self.out_frames, latent_vector_mode=self.latent_tensor_mode, latent_vector_size=self.latent_tensor_size)
        
        self.predictor = SwinSubBlock(in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                                        drop=drop, drop_path=drop_path)

        # Now create the target encoder that has a momentum average of the weights of the input encoder
        # self.target_encoder = MovingMNISTJEPAEncoder(in_channels=C, out_channels=configs['embed_dim'])
        # Copy the weights of the input encoder to the target encoder
        # self.target_encoder.load_state_dict(self.input_encoder.state_dict())
        # Make the target encoder not trainable
        # for param in self.target_encoder.parameters():
        #     param.requires_grad = False
        # # How to update the target encoder
        # self.target_encoder_update_rate = 0.999

        if configs['train_decoder']:
            # Now create a decoder that will take the output of the predictor to generate the next frames
            self.decoder = MovingMNISTJEPADecoder(in_channels=configs['embed_dim'], image_size=H)
            self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=configs['lrt_decoder'])
        else:
            self.decoder = None




    # def forward(self, frames_tensor, latent_tensor, **kwargs):
    #     # Get first 3 frames from the input'
    #     x = frames_tensor[:, :self.in_frames]
    #     # print("x shape:", x.shape)
    #     # Encode the input frames
    #     hx = self.input_encoder(x)

    #     # Encode the target frames
    #     y = frames_tensor[:, self.in_frames:self.in_frames+self.out_frames]
    #     # print("y shape:", y.shape)
    #     hy = self.input_encoder(y)

    #     # Run the ISTA algorithm to get the latent tensor
    #     if self.latent_tensor_mode == 2:
    #         with torch.enable_grad():
    #             # This trains the decoder as well (probably doesn't make sense to do this, but oh well)
    #             ISTA_output = ISTA(self.predictor, hy, hx, self.configs['sparsity_reg'], self.configs['n_steps_inf'], self.configs['lrt_z'], self.configs['tolerance'], self.latent_tensor_size, FISTA=False)
    #             latent_tensor = ISTA_output['Zs']

    #     # Enable the gradients for the predictor
    #     self.predictor.requires_grad_(True)
    #     self.predictor.train()
    #     # Predict the next frames with the latent tensor
    #     hy_hat = self.predictor(hx, latent_tensor)

    #     # Calculate all the losses
    #     h_full = torch.cat([hx, hy_hat], dim=1) # Concatenate the hidden states along the time dimension

    #     # Compute the l_vcr term
    #     l_vcr_term = l_vcr(h_full, self.configs['alpha'], self.configs['beta'])
    #     # print("l_vcr_term:", l_vcr_term)

    #     # Compute the prediction error
    #     prediction_error = torch.mean((hy_hat - hy)**2)
    #     # print("prediction_error:", prediction_error)

    #     # Compute the reconstruction error
    #     if self.configs['train_decoder']:
    #         y_pred = self.decoder(hy_hat)
    #         reconstruction_error = torch.mean((y_pred - y)**2)
    #     else:
    #         y_pred = y
    #         reconstruction_error = 0
        
    #     # print("reconstruction_error:", reconstruction_error)

    #     # Compute the total loss
    #     total_loss = prediction_error + l_vcr_term + reconstruction_error


    #     return y_pred, total_loss


class Complicated_JEPA_Model(nn.Module):
    r"""Complicated JEPA Model

    

    """

    def __init__(self, in_shape, configs, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True,  **kwargs):
        super(Complicated_JEPA_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False

        self.in_frames = configs['pre_seq_length']
        self.out_frames = configs['aft_seq_length']

        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        # model_type = 'gsta' if model_type is None else model_type.lower()
        model_type='swin'

        self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, frames_tensor, **kwargs):
        x_raw_input = frames_tensor[:, :self.in_frames]
        x_raw_target = frames_tensor[:, self.in_frames:self.in_frames+self.out_frames]

        B, T, C, H, W = x_raw_input.shape
        x_in = x_raw_input.view(B*T, C, H, W)

        B, T, C, H, W = x_raw_target.shape
        x_target = x_raw_target.view(B*T, C, H, W)

        # Encode the input frames
        embed, skip = self.enc(x_in)
        _, C_, H_, W_ = embed.shape

        # Predict the next frames
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_) # Squeeze the bactch and time dimensions

        # Encode the target frames
        target_embed, skip = self.enc(x_target)
        _, C_, H_, W_ = target_embed.shape
        z_target = target_embed.view(B, T, C_, H_, W_)


        h_full = torch.cat([z, z_target], dim=1) # Concatenate the hidden states along the time dimension
        l_vcr_term = l_vcr(h_full, self.configs['alpha'], self.configs['beta'])

        prediction_error = torch.mean((target_embed - hid)**2)


        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)

        decoder_errror = torch.mean((Y - x_target)**2)
        return Y, l_vcr_term + prediction_error + decoder_errror
    


# Test that the model runs
if __name__ == "__main__":
    # Create a simple model
    configs = {
        'in_shape': (15, 1, 64, 64), # (T, C, H, W)
        'embed_dim': 64,
        'latent_tensor_mode': 2, # ISTA
        'latent_tensor_size': 20,
        'sparsity_reg': 0.2, # Sparsity regularization parameter from paper 
        'n_steps_inf': 10,
        'lrt_z': 1,
        'tolerance': 1e-6,
        'alpha': 0.1,
        'beta': 0.1,
        'train_decoder': True,
        'lrt_decoder': 0.01
    }
    model = SimpleJEPA_Model(configs=configs)
    # Create some dummy data
    frames_tensor = torch.randn(2, 15, 1, 64, 64) # (B, T, C, H, W)
    latent_tensor = torch.randn(2, 20) # (B, latent_tensor_size) Doesn't matter what the size is for mode 2
    # mask_true = torch.randn(2, 5, 1, 64, 64)
    # Run the model
    next_frames, total_loss = model(frames_tensor, latent_tensor)
    print("Next frames shape:", next_frames.shape)
    print("Total loss:", total_loss)
    print("Model ran successfully!")

