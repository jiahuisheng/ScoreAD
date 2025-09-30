import torch

# ---
# Basic Settings
# ---
SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---
# General Dataset Settings
# ---
VARIABLE_NAME = "data" # The variable name inside the .mat file holding the data cube.

# ---
# General Training Parameters
# ---
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
# RETAIN_RATIO = 1.0 # For robust loss function

# ---
# General Diffusion Model Parameters
# ---
SIGMA = 5.0 # A suitable value for normalized data

# ---
# General Sampling & Evaluation Parameters
# ---
SAMPLE_BATCH_SIZE = 10
SAMPLER_NUM_STEPS = 1000
SCORE_PERTURBED_NUM = 100 # K value for the estimate_score function