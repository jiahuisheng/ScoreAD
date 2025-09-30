import torch
from torch.utils.data import DataLoader, TensorDataset
import functools
import scipy.io as sio
import os
import time

# Import modules and configurations from our custom files
import config as cfg
from model import ScoreModel
from utils import (
    set_seed,
    load_mat_data,
    extract_center_and_context,
    marginal_prob_std,
    diffusion_coeff,
    train,
    pc_sampler,
    ode_sampler,
    plot_samples,
    estimate_score,
)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_dataset_specific_configs(dataset_name):
    """
    Returns experiment-specific configurations based on the dataset name.
    This is the central place for managing different experiment recipes.
    """
    if dataset_name == "HYDICE_norm":
        return {
            "PATCH_SIZE": 5,
            "INNER_SIZE": 3,
            "NUM_EPOCHS": 100,
            "USE_CONDITION": False,
            "SCORE_T_VALUES": [0.05]
        }
    elif dataset_name == "Pavia_norm":
        return {
            "PATCH_SIZE": 7,
            "INNER_SIZE": 5,
            "NUM_EPOCHS": 20,
            "USE_CONDITION": True,
            "SCORE_T_VALUES": [0.05]
        }
    elif dataset_name == "Hyperion_norm":
        return {
            "PATCH_SIZE": 5,
            "INNER_SIZE": 3,
            "NUM_EPOCHS": 100,
            "USE_CONDITION": False,
            "SCORE_T_VALUES": [0.01]
        }
    elif dataset_name == "Salinas1_norm":
        return {
            "PATCH_SIZE": 3,
            "INNER_SIZE": 1,
            "NUM_EPOCHS": 100,
            "USE_CONDITION": True,
            "SCORE_T_VALUES": [0.05]
        }
    else:
        raise ValueError(f"Configuration for dataset '{dataset_name}' is not defined!")

def run_experiment(dataset_name):
    """Main function to orchestrate and run the entire experiment."""
    start_time = time.time()
    # --- Part 1: SETUP ---
    set_seed(cfg.SEED)
    
    # --- Part 2: DYNAMIC CONFIGURATION ---
    print(f"--- 0. Loading Configuration for Dataset: {dataset_name} ---")
    
    specific_configs = get_dataset_specific_configs(dataset_name)
    patch_size = specific_configs["PATCH_SIZE"]
    inner_size = specific_configs["INNER_SIZE"]
    num_epochs = specific_configs["NUM_EPOCHS"]
    use_condition = specific_configs["USE_CONDITION"]
    score_t_values = specific_configs["SCORE_T_VALUES"]

    print(f"   - Patch Size: {patch_size}, Inner Size: {inner_size}")
    print(f"   - Epochs: {num_epochs}, Use Condition: {use_condition}")
    
    # --- Part 3: DATA LOADING ---
    print("--- 1. Loading and Preparing Data ---")
    file_path = f"datasets/{dataset_name}.mat"
    data = load_mat_data(file_path, cfg.VARIABLE_NAME)
    N, M, C = data.shape
    center_x, neighbor_cond = extract_center_and_context(data, patch_size=patch_size, inner_size=inner_size)
    
    x_tensor = torch.tensor(center_x, dtype=torch.float32)
    cond_tensor = torch.tensor(neighbor_cond, dtype=torch.float32)
    
    dataset = TensorDataset(x_tensor, cond_tensor)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0) # Set num_workers=0 for simplicity
    
    print(f"Data loaded. Samples: {len(dataset)}, Spectral channels: {C}")

    # --- Part 4: MODEL INITIALIZATION ---
    print("--- 2. Initializing Model and Optimizer ---")
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=cfg.SIGMA)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=cfg.SIGMA)
    
    model = ScoreModel(
        cond_num=(patch_size**2 - inner_size**2), 
        input_dim=C, 
        marginal_prob_std=marginal_prob_std_fn
    ).to(cfg.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # --- Part 5: TRAINING ---
    print(f"--- 3. Starting Training on {cfg.DEVICE} for {num_epochs} epochs ---")
    train(
        model, 
        cfg.DEVICE, 
        dataloader, 
        optimizer, 
        marginal_prob_std_fn, 
        num_epochs=num_epochs,
        use_condition=use_condition
    )
    
    # # --- Part 6: SAMPLING & GENERATION ---
    # print("--- 4. Generating Samples ---")
    # sample_idx = torch.randint(0, len(cond_tensor), (cfg.SAMPLE_BATCH_SIZE,))
    # cond_sample = cond_tensor[sample_idx].to(cfg.DEVICE)
    
    # samples = pc_sampler(
    #     model, cond_sample, use_condition, marginal_prob_std_fn,
    #     diffusion_coeff_fn, C, batch_size=cfg.SAMPLE_BATCH_SIZE,
    #     num_steps=cfg.SAMPLER_NUM_STEPS, device=cfg.DEVICE
    # )
    # plot_samples(samples, num_samples=5)

    # --- Part 7: SCORE ESTIMATION & ANALYSIS ---
    print("--- 5. Estimating and Saving Score Maps ---")
    tensor_data = torch.tensor(data.reshape(-1, C), dtype=torch.float32)
    # Ensure results directory exists
    os.makedirs("./results", exist_ok=True)
    for val in score_t_values:
        t = torch.full((N*M,), fill_value=val, device=cfg.DEVICE)
        score = estimate_score(
            model, cfg.DEVICE, tensor_data, t, cond_tensor, use_condition, 
            marginal_prob_std_fn, cfg.SCORE_PERTURBED_NUM
        )
        score_magnitude = torch.norm(score, dim=1).view(N, M).cpu().numpy()
        
        save_path = f'./results/{dataset_name}_SAD_{use_condition}_t{val:.2f}.mat'
        timing_result = time.time() - start_time
        print(f"Saving score map to {save_path}")
        sio.savemat(save_path, {'SAD_result': score_magnitude, 'timing_results_seconds': timing_result})

if __name__ == '__main__':
    # datasets = ["HYDICE_norm", "Pavia_norm","Hyperion_norm", "Salinas1_norm"]
    datasets = ["Hyperion_norm"]
    for dataset in datasets:
        run_experiment(dataset)