import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from VMP import WalkingVAE
from utils import load_walking_data, WalkingMotionDataset, plot_full_reconstruction, plot_latent_pca, plot_joint_comparison

# Configuration
WINDOW_SIZE = 15
LATENT_DIM = 64
SELECTED_JOINTS = [
    "right_hip_pitch",
    "left_hip_pitch",
    "right_hip_yaw",
    "left_hip_yaw",
    "right_hip_roll",
    "left_hip_roll",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
]
beta = 0.05
run = 5
DATA_DIR = "data"
MODEL_DIR = f"models/ws{WINDOW_SIZE}_ld{LATENT_DIM}_beta{beta}_run{run}"
RESULTS_DIR = f"results/ws{WINDOW_SIZE}_ld{LATENT_DIM}_beta{beta}_run{run}_test"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load model and scaler ====================================================
model = WalkingVAE(input_dim=len(SELECTED_JOINTS)*2, 
                  window_size=WINDOW_SIZE, 
                  latent_dim=LATENT_DIM)

# Load pretrained weights
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pth")))
model.eval()  # Set to evaluation mode

# Load normalization parameters
scaler_params = np.load(os.path.join(MODEL_DIR, "scaler.npy"))
scaler = RobustScaler()
scaler.center_, scaler.scale_ = scaler_params[0], scaler_params[1]

# 2. Load and prepare test data ===============================================
test_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)][-2:]  # Last 2 files
test_windows = load_walking_data(test_files, WINDOW_SIZE)

# Normalize test data using training scaler
test_data = scaler.transform(
    test_windows.reshape(-1, test_windows.shape[-1])
).reshape(test_windows.shape)

test_dataset = WalkingMotionDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=256)

# 3. Calculate test loss ======================================================
def calculate_loss(batch, recon, mu, logvar):
    # Reimplementation of training loss
    # Weighted reconstruction loss
    weights = model.joint_weights.to(batch.device).unsqueeze(0)
    recon_loss = ((batch - recon).pow(2) * weights).mean()
    
    # KL divergence
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_div

total_test_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        recon, mu, logvar = model(batch)
        loss = calculate_loss(batch, recon, mu, logvar)
        total_test_loss += loss.item()

print(f"Average test loss: {total_test_loss / len(test_loader):.4f}")

# 4. Generate visualizations ==================================================
sample_idx = 42  # Fixed sample for consistency
sample = test_dataset[sample_idx].unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    recon, _, _ = model(sample)
    
    # Convert to numpy and remove batch dimension
    sample_np = sample.numpy().squeeze(0)
    recon_np = recon.numpy().squeeze(0)
    
    # Inverse normalization
    orig = scaler.inverse_transform(sample_np)
    recon = scaler.inverse_transform(recon_np)

    # Save plots
    plot_full_reconstruction(
        orig, recon, np.arange(2*WINDOW_SIZE+1),
        os.path.join(RESULTS_DIR, "full_reconstruction_test.png")
    )
    
    plot_joint_comparison(
        orig.reshape(-1, len(SELECTED_JOINTS)*2),
        recon.reshape(-1, len(SELECTED_JOINTS)*2),
        SELECTED_JOINTS,
        plot_type="position",
        save_path=os.path.join(RESULTS_DIR, "joint_positions_test.png")
    )
    
    plot_joint_comparison(
        orig.reshape(-1, len(SELECTED_JOINTS)*2),
        recon.reshape(-1, len(SELECTED_JOINTS)*2),
        SELECTED_JOINTS,
        plot_type="velocity",
        save_path=os.path.join(RESULTS_DIR, "joint_velocities_test.png")
    )

# 5. Latent space visualization ==============================================
plot_latent_pca(
    model, 
    test_loader,
    save_path=os.path.join(RESULTS_DIR, "latent_space_pca.png")
)