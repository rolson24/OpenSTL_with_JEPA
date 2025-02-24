# FISTA Algorithm implementation by kevtimova

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss term for l_vcr
def l_vcr(h):
    # h is a tensor of shape (B, T, d)
    B, T, d = h.shape
    # First we compute the varaiance term
    # 1/(T*d) * sum_t sum_i max(0, thresh - sqrt(Var(H_t_i + epsilon)))
    epsilon = 1e-6
    thresh = 1
    mean_h = torch.mean(h, dim=)


# Loss function for the ISTA algorithm
def loss_f(Zs, predictor, hx, hy, sparsity_reg, variance_reg, decoder, y, train_decoder):
    # First we compute the hinge loss term
    # 1/(T*d) * sum_t sum_i max(0, thresh - sqrt(Var(Hy_t_i + epsilon)))
    B, T_out, d = hy.shape
    T_in = hx.shape[1] # Number of input frames
    epsilon = 1e-6
    
    # Variance of the predicted frames is computed as follows:
    mean_hy = torch.mean(hy, dim=0)  # Take the mean over the batch dimension so that we get a tensor of shape (T_out, d)
    var_hy = torch.mean((hy - mean_hy)**2, dim=0)  # Take the mean over the batch dimension so that we get a tensor of shape (T_out, d)
    var_hy = torch.sqrt(var_hy + epsilon)  # Take the square root to get the standard deviation

    # Compute the hinge loss
    thresh = 1
    hinge_loss = torch.max(thresh - var_hy, torch.zeros_like(var_hy))
    hinge_loss = torch.sum(hinge_loss) / (T_out * d)

    # Next we compute the covariance term:
    # It sums the squares of the non-diagonal elements of the covariance matrix of the predicted frames.
    # A single frames covariance matrix is computed as follows:
    # N = B*d
    # cov(Hy_t) = 1/(N - 1) sum_n (Hy_t_n - mean(Hy_t)) (Hy_t_n - mean(Hy_t))^T
    frame_means = torch.mean(hy, dim=(0,2))  # shape: (T_out)
    # The covariance term is then computed as follows:
    # 1/(T*d) * sum_t sum_(i/=j) (cov(Hy_t:))|^2_(i,j)

# The ISTA algorithm is a simple iterative algorithm for solving the LASSO problem.
def ISTA(predictor, hy, hx, sparsity_reg, n_steps_inf, lrt_z, variance_reg, tolerance, hinge, FISTA=False, train_decoder=False, decoder=None, y=None,decoder_opt=None):
    """
    Args:
        predictor: the predictor network
        hy: the target vector that the predictor network should output
        hx: the hidden state from the encoder to be the input to the predictor
        sparsity_reg: the sparsity regularization parameter
        n_steps_inf: the number of steps to run the inference for
        lrt_z: the learning rate for the z variable
        variance_reg: the variance regularization parameter
        tolerance: the early stopping tolerance
        hinge: whether to use the hinge loss
        FISTA: whether to use the FISTA algorithm
        training_decoder: whether to train the decoder
        decoder: the decoder network (optional)
        y: the target frames to reproduce (optional)
        decoder_opt: the optimizer for the decoder network (optional)
    """
    # Housekeeping
    B, T_in, d = hx.shape
    T_out = hy.shape[1]
    out_channels = hy.shape[2]
    out_frames = hy.shape[1]
    latent_dim = 20

    # Turn off gradients for the predictor
    predictor.requires_grad_(False)
    predictor.eval()

    # Generate codes (initially zeros)
    Zs = nn.Parameter(torch.zeros(B, latent_dim))
    Zs.requires_grad_(True)

    if FISTA:
        aux = Zs.clone().detach()
        t_old = 1

    # Auxiliary variables for early stopping
    stop_early_dummies = torch.zeros((B, 1), device=Zs.device)
    stop_early_step = torch.zeros((B, 1), device=Zs.device)

    # Inference loop
    for step in range(n_steps_inf):
        trainable_parameters = aux if FISTA else Zs
        loss_dict = loss_f(trainable_parameters, predictor, hx, hy, sparsity_reg, variance_reg, decoder, y, train_decoder)
