import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from VMP import WalkingVAE
from utils import load_walking_data, WalkingMotionDataset, plot_full_reconstruction, plot_latent_pca, plot_joint_comparison
import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description='Train VMP model')
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--beta', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--run', type=int, default=0)
args = parser.parse_args()

# Configuration
WINDOW_SIZE = args.window_size
LATENT_DIM = args.latent_dim
BATCH_SIZE = 512
EPOCHS = 1000
BETA = args.beta
DATA_DIR = "data"
MODEL_DIR = f"models/ws{WINDOW_SIZE}_ld{LATENT_DIM}_beta{BETA}_run{args.run}"
RESULTS_DIR = f"results/ws{WINDOW_SIZE}_ld{LATENT_DIM}_beta{BETA}_run{args.run}"
SELECTED_JOINTS = [
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee',
    'right_ankle', 'left_ankle'
]
# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
train_files = files[:-2]  # Hold out 2 files for testing
train_windows = load_walking_data(train_files, WINDOW_SIZE)

# Normalization
scaler = RobustScaler()
windows_flat = train_windows.reshape(-1, train_windows.shape[-1])
scaler.fit(windows_flat)
np.save(os.path.join(MODEL_DIR, "scaler.npy"), np.array([scaler.center_, scaler.scale_]))

# Dataset and dataloader
dataset = WalkingMotionDataset(scaler.transform(windows_flat).reshape(train_windows.shape))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = WalkingVAE(input_dim=len(SELECTED_JOINTS)*2, window_size=WINDOW_SIZE, latent_dim=LATENT_DIM)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        #print(f"Batch shape: {batch.shape}")
        recon, mu, logvar = model(batch)
        
        # Weighted reconstruction loss
        weights = model.joint_weights.to(batch.device).unsqueeze(0)  # (1, 21, 20)
    
        # Calculate weighted reconstruction loss
        recon_loss = (batch - recon).pow(2) * weights
        recon_loss = recon_loss.mean()
        # recon_loss = (batch - recon).pow(2) * model.joint_weights.to(batch.device)
        # recon_loss = recon_loss.mean()
        
        # KL divergence
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + BETA * kl_div
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    train_losses.append(total_loss / len(dataloader))
    # Save visualizations every 50 epochs
    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            # Get random sample and ensure batch dimension
            sample_idx = 42
            #sample_idx = np.random.randint(len(dataset))
            sample = dataset[sample_idx].unsqueeze(0)  # Add batch dim: (1, 21, 20)
            recon, _, _ = model(sample)
            
            # Remove batch dimension safely
            sample_np = sample.numpy().squeeze(0)  # (21, 20)
            recon_np = recon.numpy().squeeze(0)     # (21, 20)
            
            # Verify shapes before inverse transform
            # assert sample_np.shape == (21, 20), f"Bad sample shape: {sample_np.shape}"
            # assert recon_np.shape == (21, 20), f"Bad recon shape: {recon_np.shape}"
            
            # Unnormalize
            orig = scaler.inverse_transform(sample_np)
            recon = scaler.inverse_transform(recon_np)
            plot_full_reconstruction(orig, recon, np.arange(2*WINDOW_SIZE+1), 
                                   os.path.join(RESULTS_DIR, f"positions_epoch_{epoch+1}.png"))
            # Plot positions
            plot_joint_comparison(
                orig.reshape(-1, len(SELECTED_JOINTS)*2),
                recon.reshape(-1, len(SELECTED_JOINTS)*2),
                SELECTED_JOINTS,
                plot_type="position",
                save_path=os.path.join(RESULTS_DIR, f"single_positions_epoch_{epoch+1}.png")
            )

            # Plot velocities (same fix)
            plot_joint_comparison(
                orig.reshape(-1, len(SELECTED_JOINTS)*2),
                recon.reshape(-1, len(SELECTED_JOINTS)*2),
                SELECTED_JOINTS,
                plot_type="velocity",
                save_path=os.path.join(RESULTS_DIR, f"single_velocities_epoch_{epoch+1}.png")
            )

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.2f}")

np.save(os.path.join(MODEL_DIR, "train_losses.npy"), np.array(train_losses))
# Save final model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.savefig(os.path.join(MODEL_DIR, "training_curve.png"))
plt.close()