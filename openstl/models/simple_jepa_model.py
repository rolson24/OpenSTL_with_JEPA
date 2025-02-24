import torch
import torch.nn as nn
import torch.nn.functional as F

from openstl.modules import MovingMNISTJEPAEncoder, MovingMNISTJEPAPredictor, MovingMNISTJEPADecoder
from openstl.utils.ISTA import ISTA

# Loss term for l_vcr (from VJ-VCR Paper by K. Drozdov et al.)
def l_vcr(h, alpha, beta):
    """
    Args:
        h: the hidden state tensor of shape (B, T, d)
        alpha: the weight of the variance term
        beta: the weight of the covariance term
    """
    # h is a tensor of shape (B, T, d)
    B, T, d = h.shape
    # First we compute the varaiance term
    # 1/(T*d) * sum_t sum_i max(0, thresh - sqrt(Var(H_t_i + epsilon)))
    epsilon = 1e-6
    thresh = 1
    mean_h = torch.mean(h, dim=0) # Take the mean over the batch dimension so that we get a tensor of shape (T, d)
    var_h = torch.mean((h - mean_h)**2, dim=0) # Take the mean over the batch dimension so that we get a tensor of shape (T, d)
    var_h = torch.sqrt(var_h + epsilon) # Take the square root to get the standard deviation
    var_term = torch.max(thresh - var_h, torch.zeros_like(var_h))
    var_term = torch.sum(var_term) / (T * d)

    # Next we compute the covariance term
    # It sums the squares of the non-diagonal elements of the covariance matrix of the frames
    # A single frames covariance matrix is computed as follows:
    # N = B*d
    # cov(H_t) = 1/(N - 1) sum_n (H_t_n - mean(H_t)) (H_t_n - mean(H_t))^T
    frame_means = torch.mean(h, dim=(0,2)) # shape: (T)
    frame_covs = 1/(B*d - 1) * torch.einsum('btd,btd->td', h - frame_means, h - frame_means) # shape: (T, d, d)
    # The covariance term is then computed as follows:
    # 1/(T*d) * sum_t sum_(i/=j) (cov(H_t))|^2_(i,j)
    cov_term = torch.sum((frame_covs * (1 - torch.eye(d, device=frame_covs.device)).unsqueeze(0))**2) / (T * d)

    return alpha*var_term + beta*cov_term

class SimpleJEPA_Model(nn.Module):
    r"""SimpleJEPA Model

    Implementation of `Video Representation Learning with Joint-Embedding Predictive Architectures <https://arxiv.org/pdf/2412.10925>`_.

    """
    def __init__(self, configs, **kwargs):
        super(SimpleJEPA_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs

        self.latent_tensor_mode = configs.latent_tensor_mode
        self.latent_tensor_size = configs.latent_tensor_size

        self.input_encoder = MovingMNISTJEPAEncoder(in_channels=C, out_channels=configs.embed_dim)

        self.predictor = MovingMNISTJEPAPredictor(in_channels=configs.embed_dim, out_channels=configs.embed_dim, latent_vector_mode=self.latent_tensor_mode, latent_vector_size=self.latent_tensor_size)

        # Now create the target encoder that has a momentum average of the weights of the input encoder
        self.target_encoder = MovingMNISTJEPAEncoder(in_channels=C, out_channels=configs.embed_dim)
        # Copy the weights of the input encoder to the target encoder
        self.target_encoder.load_state_dict(self.input_encoder.state_dict())
        # Make the target encoder not trainable
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        # How to update the target encoder
        self.target_encoder_update_rate = 0.999

        if configs.train_decoder:
            # Now create a decoder that will take the output of the predictor to generate the next frames
            self.decoder = MovingMNISTJEPADecoder(in_channels=configs.embed_dim, image_size=H)
            self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=configs.lrt_decoder)
        else:
            self.decoder = None




    def forward(self, frames_tensor, latent_tensor, mask_true, **kwargs):
        # Get first 3 frames from the input'
        x = frames_tensor[:, :3]
        # Encode the input frames
        hx = self.input_encoder(x)

        # Encode the target frames
        y = frames_tensor[:, 3:]
        hy = self.target_encoder(y)


        # Predict the next frames with the latent tensor
        if self.latent_tensor_mode == 'ISTA':
            # This trains the decoder as well (probably doesn't make sense to do this, but oh well)
            latent_tensor = ISTA(self.predictor, hx, latent_tensor, self.configs.sparsity_reg, self.configs.n_steps_inf, self.configs.lrt_z, self.configs.tolerance, self.latent_tensor_size, self.configs.FISTA, self.configs.train_decoder, self.decoder, y, self.decoder_opt)
        
        hy_hat = self.predictor(hx, latent_tensor)



        # Calculate all the losses
        h_full = torch.cat([hx, hy_hat], dim=1) # Concatenate the hidden states along the time dimension
        l_vcr_term = l_vcr(h_full, self.configs.alpha, self.configs.beta)

        # Compute the prediction error
        prediction_error = torch.mean((hy_hat - hy)**2)

        total_loss = prediction_error + l_vcr_term

        # Need to modify to compute the decoder outside of ISTA
        next_frames = None
        return next_frames, total_loss


# Test that the model runs
if __name__ == "__main__":
    # Create a simple model
    model = SimpleJEPA_Model(configs=None)
    # Create some dummy data
    frames_tensor = torch.randn(2, 5, 1, 64, 64)
    latent_tensor = torch.randn(2, 64)
    mask_true = torch.randn(2, 5, 1, 64, 64)
    # Run the model
    next_frames, total_loss = model(frames_tensor, latent_tensor, mask_true)
    print("Next frames shape:", next_frames.shape)
    print("Total loss:", total_loss)
    print("Model ran successfully!")