import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import os
from torch.utils.data import Dataset, DataLoader

# WINDOW_SIZE = 10  # 21 frames (0.35s @60Hz)
# LATENT_DIM = 16
# BATCH_SIZE = 256
# EPOCHS = 1000
# BETA = 0.05
SELECTED_JOINTS = [
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee',
    'right_ankle', 'left_ankle'
]

def load_walking_data(file_paths, window_size):
    all_frames = []
    for path in file_paths:
        # Check if file exists
        if not os.path.exists(path):
            print(f"File {path} not found!")
            continue
            
        df = pd.read_csv(path)
        
        # Verify required columns
        if 'joint_name' not in df.columns:
            print(f"File {path} missing 'joint_name' column")
            continue
            
        # Filter joints
        df = df[df['joint_name'].isin(SELECTED_JOINTS)]
        if df.empty:
            print(f"No valid joints found in {path}")
            continue
            
        # Pivot and check
        try:
            pivoted = df.pivot(index='elapsed_time', 
                              columns='joint_name',
                              values=['position', 'velocity'])
            
            pivoted.columns = [f"{j}_{typ}" for typ,j in pivoted.columns]
            pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)
            frames = pivoted.dropna().values.tolist()
            if not frames:
                print(f"No valid frames in {path}")
                continue
            all_frames.extend(frames)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Create windows with validation
    windows = []
    n_frames = len(all_frames)
    required_length = 2 * window_size + 1
    
    if n_frames < required_length:
        print(f"Not enough frames: {n_frames} < {required_length}")
        return np.array([])
    
    for i in range(n_frames - required_length + 1):
        window = np.array(all_frames[i:i+required_length])
        # Check ankle columns (last 4 features)
        #if window[:, -4:].mean() > 0.01:  # Reduced threshold
        windows.append(window)
    
    print(f"Created {len(windows)} valid windows")
    return np.array(windows) if windows else np.array([])


class WalkingMotionDataset(Dataset):
    def __init__(self, windows):
        self.windows = torch.FloatTensor(windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx]
    

def plot_joint_comparison(original, reconstructed, joint_names, plot_type="position", save_path=None):
    plt.figure(figsize=(15, 10))
    n_joints = len(joint_names)
    
    for i, joint in enumerate(joint_names):
        plt.subplot(n_joints//2, 2, i+1)
        
        if plot_type == "position":
            orig = original[:, i]
            recon = reconstructed[:, i]
            ylabel = "Position (rad)"
        else:
            orig = original[:, i + len(joint_names)]
            recon = reconstructed[:, i + len(joint_names)]
            ylabel = "Velocity (rad/s)"
        
        plt.plot(orig, label='Original', linewidth=2)
        plt.plot(recon, label='Reconstructed', linestyle='--')
        plt.title(f"{joint} {plot_type.capitalize()}")
        plt.ylabel(ylabel)
        plt.xlabel("Frame")
        plt.grid(True)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_full_reconstruction(orig, recon, timesteps, save_path):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    for j in range(len(SELECTED_JOINTS)):
        axs[0].plot(timesteps, orig[:, j], label=f'{SELECTED_JOINTS[j]} (orig)', linestyle='-')
        axs[0].plot(timesteps, recon[:, j], label=f'{SELECTED_JOINTS[j]} (recon)', linestyle='--')
    axs[0].set_title('Joint Positions')
    axs[0].legend()
    
    for j in range(len(SELECTED_JOINTS)):
        axs[1].plot(timesteps, orig[:, j + len(SELECTED_JOINTS)], linestyle='-')
        axs[1].plot(timesteps, recon[:, j + len(SELECTED_JOINTS)], linestyle='--')
    axs[1].set_title('Joint Velocities')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_latent_pca(model, dataloader, save_path):
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            mu, _ = model.encode(batch)
            latents.append(mu.numpy())
    latents = np.concatenate(latents)
    
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latents)
    
    plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Latent Space PCA')
    plt.savefig(save_path)
    plt.close()