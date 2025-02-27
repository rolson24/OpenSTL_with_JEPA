import torch
import torch.nn as nn
import torch.nn.functional as F

# from openstl.modules import MovingMNISTJEPAEncoder, MovingMNISTJEPAPredictor, MovingMNISTJEPADecoder
from openstl.modules import Backbone_VMAMBA2, VMAMBA2_Video_Encoder, MovingMNISTJEPAPredictor, JEPAImageDecoder, JEPAConvImageDecoder
from openstl.utils.ISTA import ISTA


def l_vcr(h, alpha, beta):
    """
    Args:
        h: the hidden state tensor of shape (B, T, d)
        alpha: the weight of the variance term
        beta: the weight of the covariance term
    """
    # print("h: ", h)
    B, T, d = h.shape
    # print("min: ", torch.min(h))
    # print("max: ", torch.max(h))
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

class JEPA_Mamba_Model(nn.Module):
    r"""SimpleJEPA Model

    Implementation of `Video Representation Learning with Joint-Embedding Predictive Architectures <https://arxiv.org/pdf/2412.10925>`.
    Modified to use the VMAMBA2 backbone from `VSSD: Vision Mamba with Non-Causal State Space Duality <https://arxiv.org/abs/2407.18559>` for the encoder.

    """
    def __init__(self, configs, **kwargs):
        super(JEPA_Mamba_Model, self).__init__()
        T, C, H, W = configs['in_shape']

        self.configs = configs

        self.latent_tensor_mode = configs['latent_tensor_mode']
        self.latent_tensor_size = configs['latent_tensor_size']

        self.in_frames = configs['pre_seq_length']
        self.out_frames = configs['aft_seq_length']
        
        self.embed_dim = configs['embed_dim']

        self.encoder = VMAMBA2_Video_Encoder(
            image_size=H,
            patch_size=configs['patch_size'],
            in_chans=C,
            embed_dim=self.embed_dim,
            depths=configs['depths'],
            num_heads=configs['num_heads'],
            mlp_ratio=configs['mlp_ratio'],
            qkv_bias=configs['qkv_bias'],
            drop_rate=configs['drop_rate'],
            drop_path_rate=configs['drop_path_rate'],
            ssd_expansion=configs['ssd_expansion'],
            ssd_ngroups=configs['ssd_ngroups'],
            ssd_chunk_size=configs['ssd_chunk_size'],
            linear_attn_duality=configs['linear_attn_duality'],
            d_state=configs['d_state']
        )

        self.encoding_dim = self.embed_dim * 8 # ???


        self.predictor = MovingMNISTJEPAPredictor(in_channels=self.encoding_dim, out_channels=self.encoding_dim, in_frames=self.in_frames, out_frames=self.out_frames, latent_vector_mode=self.latent_tensor_mode, latent_vector_size=self.latent_tensor_size)



        if configs['train_decoder']:
            # Now create a decoder that will take the output of the predictor to generate the next frames
            self.decoder = JEPAImageDecoder(in_channels=self.encoding_dim, img_channels=C, img_height=H, img_width=W)
            self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=configs['lrt_decoder'])
        else:
            self.decoder = None




    def forward(self, frames_tensor, latent_tensor, **kwargs):
        # Get first 3 frames from the input'
        x = frames_tensor[:, :self.in_frames]
        # print("x shape:", x.shape)
        # Encode the input frames
        hx = self.encoder(x)

        # Encode the target frames
        y = frames_tensor[:, self.in_frames:self.in_frames+self.out_frames]
        # print("y shape:", y.shape)
        hy = self.encoder(y)

        # Run the ISTA algorithm to get the latent tensor
        if self.latent_tensor_mode == 2:
            with torch.enable_grad():
                # This trains the decoder as well (probably doesn't make sense to do this, but oh well)
                ISTA_output = ISTA(self.predictor, hy, hx, self.configs['sparsity_reg'], self.configs['n_steps_inf'], self.configs['lrt_z'], self.configs['tolerance'], self.latent_tensor_size, FISTA=False)
                latent_tensor = ISTA_output['Zs']

        # Enable the gradients for the predictor
        self.predictor.requires_grad_(True)
        self.predictor.train()
        # Predict the next frames with the latent tensor
        hy_hat = self.predictor(hx, latent_tensor)

        # Calculate all the losses
        h_full = torch.cat([hx, hy_hat], dim=1) # Concatenate the hidden states along the time dimension

        # Compute the l_vcr term
        l_vcr_term = l_vcr(h_full, self.configs['alpha'], self.configs['beta'])
        # print("l_vcr_term:", l_vcr_term)

        # Compute the prediction error
        prediction_error = torch.mean((hy_hat - hy)**2)
        # print("prediction_error:", prediction_error)

        # Compute the reconstruction error
        if self.configs['train_decoder']:
            y_pred = self.decoder(hy_hat)
            reconstruction_error = torch.mean((y_pred - y)**2)
        else:
            y_pred = y
            reconstruction_error = 0
        
        # print("reconstruction_error:", reconstruction_error)

        # Compute the total loss
        total_loss = prediction_error + l_vcr_term + reconstruction_error


        return y_pred, total_loss


# Test that the model runs
if __name__ == "__main__":
    # Create a simple model
    configs = {
        'in_shape': [15, 1, 64, 64], # (T, C, H, W)
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
    model = JEPA_Mamba_Model(configs=configs)
    # Create some dummy data
    frames_tensor = torch.randn(2, 15, 1, 64, 64) # (B, T, C, H, W)
    latent_tensor = torch.randn(2, 20) # (B, latent_tensor_size) Doesn't matter what the size is for mode 2
    # mask_true = torch.randn(2, 5, 1, 64, 64)
    # Run the model
    next_frames, total_loss = model(frames_tensor, latent_tensor)
    print("Next frames shape:", next_frames.shape)
    print("Total loss:", total_loss)
    print("Model ran successfully!")