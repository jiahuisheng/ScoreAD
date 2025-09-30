import torch
import numpy as np
import random
import os
from scipy.io import loadmat
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed=1):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Diffusion & Loss Functions ---
def marginal_prob_std(t, sigma):
    """Computes the standard deviation of the perturbation kernel."""
    return torch.sqrt((sigma ** (2 * t) - 1.) / (2 * np.log(sigma)))

def diffusion_coeff(t, sigma):
    """Computes the diffusion coefficient of the SDE."""
    return torch.tensor(sigma**t, device=t.device)
    
def conditional_robust_loss_fn(model, x, cond, use_condition, marginal_prob_std, eps=1e-5):
    """The loss function for the conditional score model."""
    t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(t)
    perturbed_x = x + std[:, None] * z
    score = model(perturbed_x, t, cond, use_condition)
    per_sample_loss = torch.sum((score * std[:, None] + z) ** 2, dim=-1)
    # k = int(retain_ratio * per_sample_loss.shape[0])
    # sorted_loss, _ = torch.sort(per_sample_loss)
    # filtered_loss = sorted_loss[:k]
    return per_sample_loss.mean()

# --- Training & Evaluation ---
def train(model, device, dataloader, optimizer, marginal_prob_std_fn, num_epochs, use_condition):
    """The main training loop."""
    model.train()
    losses = []
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)
            cond_for_model = batch[1].to(device)
            loss = conditional_robust_loss_fn(model, x, cond_for_model, use_condition, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # # Plot loss curve
    # plt.figure(figsize=(10, 5))
    # plt.plot(losses, label='Loss', color='blue')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve During Training')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

def estimate_score(model, device, x, t, cond, use_condition, marginal_prob_std, K, eps=1e-5):
    """Estimates the score by averaging over multiple perturbations."""
    model.eval() 
    with torch.no_grad():
        t, x, cond = t.to(device), x.to(device), cond.to(device)
        total_score = torch.zeros_like(x)
        for _ in range(K):
            z = torch.randn_like(x) 
            std = marginal_prob_std(t)
            perturbed_x = x + std[:, None] * z
            score = model(perturbed_x, t, cond, use_condition)
            score_norm = score / torch.norm(score, p=2, dim=1, keepdim=True)
            total_score += score_norm
    return total_score

# --- Samplers ---
def pc_sampler(score_model, cond, use_condition, marginal_prob_std, diffusion_coeff, spectral_channels, batch_size=64, num_steps=500, snr=0.16, device='cuda', eps=1e-3):
    """Predictor-Corrector sampler."""
    t = torch.ones(batch_size, device=device)
    if cond is not None and cond.shape[0] != batch_size:
        cond = cond.repeat(batch_size // cond.shape[0], 1, 1)
    init_x = torch.randn(batch_size, spectral_channels, device=device) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="PC Sampling"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            score = score_model(x, batch_time_step, cond, use_condition)
            grad = score
            grad_norm = torch.norm(grad, dim=-1).mean()
            noise_norm = np.sqrt(spectral_channels)
            if grad_norm.item() > 1e-6:
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            else:
                langevin_step_size = 0.0
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None] * score * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(x)
    return x_mean

def ode_sampler(score_model, cond, use_condition, marginal_prob_std, diffusion_coeff, spectral_channels, batch_size=64, num_steps=1000, device='cuda', eps=1e-3):
    """ODE sampler."""
    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, spectral_channels, device=device) * marginal_prob_std(t)[:, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="ODE Sampling"):
            t_batch = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(t_batch)
            score = score_model(x, t_batch, cond, use_condition)
            x = x + (g**2)[:, None] * score * step_size
    return x
    
# --- Data Processing ---
def load_mat_data(file_path, variable_name='data'):
    """Loads a .mat file."""
    mat_data = loadmat(file_path)
    if variable_name not in mat_data:
        raise ValueError(f"Variable '{variable_name}' not found in the .mat file.")
    return mat_data[variable_name]

def extract_center_and_context(data, patch_size=5, inner_size=3):
    """Extracts center pixels and their surrounding context ring."""
    assert patch_size % 2 == 1 and inner_size % 2 == 1, "Patch sizes must be odd"
    assert patch_size > inner_size, "Outer patch must be larger than inner patch"
    H, W, C = data.shape
    pad = patch_size // 2
    inner_pad = inner_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    centers, contexts = [], []
    for i in range(pad, pad + H):
        for j in range(pad, pad + W):
            center = padded[i, j, :]
            outer_patch = padded[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            outer_mask = np.ones((patch_size, patch_size), dtype=bool)
            outer_mask[pad - inner_pad:pad + inner_pad + 1, pad - inner_pad:pad + inner_pad + 1] = False
            ring_pixels = outer_patch[outer_mask].reshape(-1, C)
            centers.append(center)
            contexts.append(ring_pixels)
    return np.stack(centers), np.stack(contexts)

# --- Plotting ---
def plot_samples(samples, num_samples=5):
    """Plots the first few generated samples."""
    num_samples = min(num_samples, samples.shape[0])
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.plot(samples[i].cpu().numpy(), label=f'Sample {i+1}')
    plt.title(f'Generated Samples (First {num_samples})')
    plt.xlabel('Band Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()