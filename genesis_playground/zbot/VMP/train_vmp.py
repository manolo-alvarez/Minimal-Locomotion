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
parser.add_argument('--window_size', type=int, default=15)
parser.add_argument('--latent_dim', type=int, default=64)
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
DATA_DIR = "genesis_playground/zbot/data"
MODEL_DIR = f"models/ws{WINDOW_SIZE}_ld{LATENT_DIM}_beta{BETA}_run{args.run}"
RESULTS_DIR = f"results/ws{WINDOW_SIZE}_ld{LATENT_DIM}_beta{BETA}_run{args.run}"
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

#Validation daa
val_files = files[-2:]  # Last two files for validation
val_windows = load_walking_data(val_files, WINDOW_SIZE)
val_windows_flat = val_windows.reshape(-1, val_windows.shape[-1])
val_dataset = WalkingMotionDataset(scaler.transform(val_windows_flat).reshape(val_windows.shape))
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = WalkingVAE(input_dim=len(SELECTED_JOINTS)*2, window_size=WINDOW_SIZE, latent_dim=LATENT_DIM)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10  # Number of epochs to wait before stopping
patience_counter = 0
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
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
        
        total_train_loss += loss.item()
    train_losses.append(total_train_loss / len(dataloader))

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            recon, mu, logvar = model(batch)
            
            weights = model.joint_weights.to(batch.device).unsqueeze(0)
            recon_loss = (batch - recon).pow(2) * weights
            recon_loss = recon_loss.mean()
            
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            val_loss = recon_loss + BETA * kl_div
            total_val_loss += val_loss.item()
    
    val_losses.append(total_val_loss / len(val_dataloader))
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

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.2f} | Val Loss: {val_losses[-1]:.2f}")
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0  # Reset patience counter
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))  # Save best model
        print(f"✅ Saved new best model with Val Loss: {best_val_loss:.2f}")
    else:
        patience_counter += 1  # Increase patience counter

    if patience_counter >= patience:
        print(f"🚀 Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.2f}")
        break  # Stop training

np.save(os.path.join(MODEL_DIR, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(MODEL_DIR, "val_losses.npy"), np.array(val_losses))

# Plot training and validation curves
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss", linestyle="dashed")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Curve")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "training_validation_curve.png"))
plt.close()