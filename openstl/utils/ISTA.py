# FISTA Algorithm implementation by kevtimova

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





# Loss function for the ISTA algorithm
def loss_f(Zs, predictor, hx, hy):
    """
    Args:
        Zs: the latent codes
        predictor: the predictor network
        hx: the hidden state from the encoder
        hy: the target vector
    """

    # Compute the prediction error
    # print("hx",hx.shape)
    # print("Zs",Zs.shape)
    h_pred = predictor(hx, Zs)
    error = torch.mean((h_pred - hy)**2)

    # if train_decoder:
    #     # Compute the reconstruction error
    #     y_pred = decoder(h_pred)
    #     reconstruction_error = torch.mean((y_pred - y)**2)
    # else:
    #     reconstruction_error = 0
    

    output = {'error': error, 'hy_hat': h_pred}
    return output


# The ISTA algorithm is a simple iterative algorithm for solving the LASSO problem.
def ISTA(predictor, hy, hx, sparsity_reg, n_steps_inf, lrt_z, tolerance, Zs_dim, FISTA=False):
    """
    Args:
        predictor: the predictor network
        hy: the target vector that the predictor network should output
        hx: the hidden state from the encoder to be the input to the predictor
        sparsity_reg: the sparsity regularization parameter
        n_steps_inf: the number of steps to run the inference for
        lrt_z: the learning rate for the z variable
        tolerance: the early stopping tolerance
        FISTA: whether to use the FISTA algorithm
    """
    # Housekeeping
    B, T_in, d = hx.shape
    T_out = hy.shape[1]
    out_channels = hy.shape[2]
    out_frames = hy.shape[1]
    latent_dim = Zs_dim

    # Turn off gradients for the predictor
    predictor.requires_grad_(False)
    predictor.eval()

    # Generate codes (initially random)
    # Zs = nn.Parameter(torch.zeros(B, latent_dim))
    Zs = nn.Parameter(torch.randn(B, latent_dim))
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
        loss_dict = loss_f(trainable_parameters, predictor, hx, hy)
        error = loss_dict['error']
        # print("error",error)

        # Gradient computation
        trainable_parameters.grad = None
        error.backward(retain_graph=True)

        # print("Zs grad",Zs.grad)
        # print("trainable_parameters grad",trainable_parameters.grad)

        # Keep track of old values for FISTA
        Zs_old = Zs.clone().detach()
        # print("Old Zs",Zs_old)

        # Gradient and shrinkage step
        Zs = ISTA_step(x=trainable_parameters, alpha=sparsity_reg, step_size=lrt_z, stop_early=stop_early_dummies, positive=True)
        # print("New Zs",Zs)

        # FISTA
        if FISTA:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old**2))
            aux = nn.Parameter(Zs.detach() + (t_old - 1) / t_new * (Zs.detach() - Zs_old))
            t_old = t_new
        
        # Early stopping
        stop_early_dummies = stop_early(Zs_old, Zs.detach(), tolerance)
        # print("stop_early_dummies",stop_early_dummies)

        # Stop early if all elements are below the tolerance
        stop_early_step += 1 - stop_early_dummies
        if step < n_steps_inf - 1 and torch.sum(stop_early_dummies) == B:
            break

    # Count num of total steps
    Zs_steps_mean = torch.mean(stop_early_step)

    # Remove gradients
    Zs = Zs.detach()
    Zs.requires_grad_(False)

    # # Train the decoder
    # if train_decoder:
    #     decoder_opt.zero_grad()
    #     loss_dict = loss_f(Zs, predictor, hx, hy, train_decoder=True, decoder=decoder, y=y)
    #     loss = loss_dict['reconstruction_error']
    #     loss.backward()
    #     decoder_opt.step()

    output = {'Zs': Zs, 'Zs_steps_mean': Zs_steps_mean, 'error': error}
    return output


def ISTA_step(x, alpha, step_size, stop_early, positive=False):
    """
    Args:
        x: the input tensor
        alpha: the regularization parameter
        step_size: the step size
        stop_early: whether to stop early
        positive: whether to enforce positivity
    """

    z_prox = x.clone().detach()
    # ISTA gradient step followed by shrinkage
    with torch.no_grad():
        z_prox.data = soft_treshold(x.detach() - (1 - stop_early) * step_size * x.grad.data,
                                    threshold=(1 - stop_early) * alpha * step_size, positive=positive)
    
    return nn.Parameter(z_prox)

def soft_treshold(x, threshold, positive=True):
    """
    Function that shirnks the input tensor by a threshold value
    Args:
        x: the input tensor
        threshold: the threshold value
        positive: whether to enforce positivity
    """
    result = x.sign() * F.relu(x.abs() - threshold, inplace=True)
    # print("result",result)
    if positive:
        result = F.relu(result, inplace=True)
    return result

def stop_early(z_old, z_new, tolerance):
    """
    Function that stops the optimization early if the difference between the old and new values is below a certain tolerance
    Args:
        x_old: the old tensor
        x_new: the new tensor
        tolerance: the tolerance value
    """
    if tolerance == 0:
        shape = (z_old.shape[0], 1) if len(z_old.shape) == 2 else (z_old.shape[0], 1, 1, 1)
        # print("shape",shape)
        return torch.zeros(shape, device=z_old.device)
    with torch.no_grad():
        code_dim = 1 if len(z_old.shape) == 2 else (1, 2, 3)
        diff = torch.norm(z_old - z_new, p=2, dim=code_dim) / torch.norm(z_old, p=2, dim=code_dim)
        # print("diff",diff)
        if len(z_old.shape) == 2:
            diff = diff.unsqueeze(1)
            return (diff < tolerance).float()
        else:
            return (diff < tolerance).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)