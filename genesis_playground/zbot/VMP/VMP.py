import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


# # Configuration
# WINDOW_SIZE = 10  # 21 frames (0.35s @60Hz)
# LATENT_DIM = 32
# BATCH_SIZE = 256
# EPOCHS = 1000
# BETA = 0.05
SELECTED_JOINTS = [
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee',
    'right_ankle', 'left_ankle'
]

# class WalkingMotionDataset(Dataset):
#     def __init__(self, windows):
#         self.windows = torch.FloatTensor(windows)
        
#     def __len__(self): return len(self.windows)
#     def __getitem__(self, idx): return self.windows[idx]
dropout = 0.2

class WalkingVAE(nn.Module):
    def __init__(self, input_dim, window_size , latent_dim ):
        super().__init__()
        self.flat_dim = (2*window_size+1) * input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        print("true dim",self.window_size, self.latent_dim)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
        )

        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, self.flat_dim))
        
        # Joint weights (prioritize knees and ankles)
                # Each joint has 2 features: [position, velocity]
        joint_weights_base = torch.tensor([
            # Left side          # Position  # Velocity
            1.0, 1.0,           # hip_yaw
            1.0, 1.0,           # hip_roll
            1.5, 1.5,           # hip_pitch
            2.0, 2.0,           # knee
            # Right side
            1.0, 1.0,           # hip_yaw
            1.0, 1.0,           # hip_roll
            1.5, 1.5,           # hip_pitch
            2.0, 2.0,           # knee
            # Ankles
            2.0, 2.0,           # right
            2.0, 2.0            # left
        ])
        
        # Repeat weights for each frame in window
        self.joint_weights = joint_weights_base.repeat(2*window_size+1, 1)

    def encode(self, x):
        #print(f"Input shape before flattening: {x.shape}")  # Debugging line
    
        x_flat = x.view(-1, self.flat_dim)
        h = self.encoder(x_flat)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10, max=10)

    def decode(self, z):
        return self.decoder(z).view(-1, 2*self.window_size+1, len(SELECTED_JOINTS)*2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

# def load_walking_data(file_paths):
#     all_frames = []
#     for path in file_paths:
#         # Check if file exists
#         if not os.path.exists(path):
#             print(f"File {path} not found!")
#             continue
            
#         df = pd.read_csv(path)
        
#         # Verify required columns
#         if 'joint_name' not in df.columns:
#             print(f"File {path} missing 'joint_name' column")
#             continue
            
#         # Filter joints
#         df = df[df['joint_name'].isin(SELECTED_JOINTS)]
#         if df.empty:
#             print(f"No valid joints found in {path}")
#             continue
            
#         # Pivot and check
#         try:
#             pivoted = df.pivot(index='elapsed_time', 
#                               columns='joint_name',
#                               values=['position', 'velocity'])
            
#             pivoted.columns = [f"{j}_{typ}" for typ,j in pivoted.columns]
#             pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)
#             frames = pivoted.dropna().values.tolist()
#             if not frames:
#                 print(f"No valid frames in {path}")
#                 continue
#             all_frames.extend(frames)
#         except Exception as e:
#             print(f"Error processing {path}: {str(e)}")
#             continue
    
#     # Create windows with validation
#     windows = []
#     n_frames = len(all_frames)
#     required_length = 2 * WINDOW_SIZE + 1
    
#     if n_frames < required_length:
#         print(f"Not enough frames: {n_frames} < {required_length}")
#         return np.array([])
    
#     for i in range(n_frames - required_length + 1):
#         window = np.array(all_frames[i:i+required_length])
#         # Check ankle columns (last 4 features)
#         #if window[:, -4:].mean() > 0.01:  # Reduced threshold
#         windows.append(window)
    
#     print(f"Created {len(windows)} valid windows")
#     return np.array(windows) if windows else np.array([])

# def plot_joint_comparison(original, reconstructed, joint_names, plot_type="position", save_path=None):
#     plt.figure(figsize=(15, 10))
#     n_joints = len(joint_names)
    
#     for i, joint in enumerate(joint_names):
#         plt.subplot(n_joints//2, 2, i+1)
        
#         if plot_type == "position":
#             orig = original[:, i]
#             recon = reconstructed[:, i]
#             ylabel = "Position (rad)"
#         else:
#             orig = original[:, i + len(joint_names)]
#             recon = reconstructed[:, i + len(joint_names)]
#             ylabel = "Velocity (rad/s)"
        
#         plt.plot(orig, label='Original', linewidth=2)
#         plt.plot(recon, label='Reconstructed', linestyle='--')
#         plt.title(f"{joint} {plot_type.capitalize()}")
#         plt.ylabel(ylabel)
#         plt.xlabel("Frame")
#         plt.grid(True)
#         if i == 0:
#             plt.legend()
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

# Training pipeline
# if __name__ == "__main__":
#     # Load and prepare data
#     files = ["data/"+f for f in os.listdir("data")]
#     windows = load_walking_data(files)
    
#     # Normalization
#     scaler = RobustScaler()
#     windows_flat = windows.reshape(-1, windows.shape[-1])
#     scaler.fit(windows_flat)
#     windows_norm = scaler.transform(windows_flat).reshape(windows.shape)
    
#     # Create dataloader
#     dataset = WalkingMotionDataset(windows_norm)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
#     # Initialize model
#     model = WalkingVAE(len(SELECTED_JOINTS)*2)  # Position + velocity
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
#     # Training loop
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
        
#         for batch in dataloader:
#             optimizer.zero_grad()
#             recon, mu, logvar = model(batch)
            
#             # Weighted reconstruction loss
#             weights = model.joint_weights.to(batch.device).unsqueeze(0)  # (1, 21, 20)
        
#             # Calculate weighted reconstruction loss
#             recon_loss = (batch - recon).pow(2) * weights
#             recon_loss = recon_loss.mean()
#             # recon_loss = (batch - recon).pow(2) * model.joint_weights.to(batch.device)
#             # recon_loss = recon_loss.mean()
            
#             # KL divergence
#             kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
#             loss = recon_loss + BETA * kl_div
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         # Save visualizations every 50 epochs
#         if (epoch+1) % 50 == 0:
#             with torch.no_grad():
#                 # Get random sample and ensure batch dimension
#                 sample_idx = 42
#                 #sample_idx = np.random.randint(len(dataset))
#                 sample = dataset[sample_idx].unsqueeze(0)  # Add batch dim: (1, 21, 20)
#                 recon, _, _ = model(sample)
                
#                 # Remove batch dimension safely
#                 sample_np = sample.numpy().squeeze(0)  # (21, 20)
#                 recon_np = recon.numpy().squeeze(0)     # (21, 20)
                
#                 # Verify shapes before inverse transform
#                 assert sample_np.shape == (21, 20), f"Bad sample shape: {sample_np.shape}"
#                 assert recon_np.shape == (21, 20), f"Bad recon shape: {recon_np.shape}"
                
#                 # Unnormalize
#                 orig = scaler.inverse_transform(sample_np)
#                 recon = scaler.inverse_transform(recon_np)

#                 # Plot positions
#                 plot_joint_comparison(
#                     orig.reshape(-1, len(SELECTED_JOINTS)*2),
#                     recon.reshape(-1, len(SELECTED_JOINTS)*2),
#                     SELECTED_JOINTS,
#                     plot_type="position",
#                     save_path=f"positions_epoch_{epoch+1}.png"
#                 )

#                 # Plot velocities (same fix)
#                 plot_joint_comparison(
#                     orig.reshape(-1, len(SELECTED_JOINTS)*2),
#                     recon.reshape(-1, len(SELECTED_JOINTS)*2),
#                     SELECTED_JOINTS,
#                     plot_type="velocity",
#                     save_path=f"velocities_epoch_{epoch+1}.png"
#                 )

#         print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.2f}")

#     # Save final model
#     torch.save(model.state_dict(), "walking_vae.pth")