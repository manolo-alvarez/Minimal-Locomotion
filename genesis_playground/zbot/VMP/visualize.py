# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# def plot_latent_space(model, dataloader, n_samples=1000):
#     """Visualize latent space using T-SNE"""
#     model.eval()
#     latents = []
    
#     with torch.no_grad():
#         for i, batch in enumerate(dataloader):
#             mu, _ = model.encode(batch)
#             latents.append(mu.numpy())
#             if i*batch.size(0) > n_samples:
#                 break
                
#     latents = np.concatenate(latents)[:n_samples]
    
#     # T-SNE projection
#     tsne = TSNE(n_components=2)
#     latent_2d = tsne.fit_transform(latents)
    
#     plt.figure(figsize=(10,8))
#     plt.scatter(latent_2d[:,0], latent_2d[:,1], alpha=0.6)
#     plt.title("Latent Space T-SNE Projection")
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()

# def plot_reconstructions(model, dataset, n_examples=5):
#     """Compare original vs reconstructed motions"""
#     model.eval()
#     indices = np.random.choice(len(dataset), n_examples)
    
#     fig, axs = plt.subplots(n_examples, 2, figsize=(12, 3*n_examples))
    
#     for i, idx in enumerate(indices):
#         original = dataset[idx].numpy()
        
#         with torch.no_grad():
#             reconstructed, _, _ = model(original[None])
#             reconstructed = reconstructed.numpy()[0]
        
#         # Plot first 3 dimensions for visualization
#         axs[i,0].plot(original[:,0], label='Original')
#         axs[i,0].plot(reconstructed[:,0], label='Reconstructed')
#         axs[i,0].set_title(f"Example {i+1} - Position")
        
#         axs[i,1].plot(original[:,1], label='Original')
#         axs[i,1].plot(reconstructed[:,1], label='Reconstructed')
#         axs[i,1].set_title(f"Example {i+1} - Velocity")
    
#     plt.tight_layout()
#     plt.legend()
#     plt.show()

# def plot_latent_traversal(model, scaler, n_steps=10):
#     """Visualize latent dimension variations"""
#     model.eval()
#     z_base = torch.zeros(1, model.latent_dim)
    
#     fig, axs = plt.subplots(model.latent_dim//4, 4, figsize=(20, 10))
    
#     for dim in range(model.latent_dim):
#         row = dim // 4
#         col = dim % 4
        
#         # Traverse latent dimension
#         z_values = torch.linspace(-3, 3, n_steps)
#         samples = []
        
#         for val in z_values:
#             z = z_base.clone()
#             z[0,dim] = val
#             with torch.no_grad():
#                 sample = model.decode(z).numpy()
#             samples.append(sample)
        
#         # Unnormalize and plot
#         samples = np.array(samples).reshape(n_steps, -1)
#         samples = scaler.inverse_transform(samples)
#         axs[row,col].plot(samples[:,0], samples[:,1], 'o-')
#         axs[row,col].set_title(f"Latent Dim {dim}")
    
#     plt.tight_layout()
#     plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from VMP import StableVAE, load_data, prepare_dataloader

def visualize_model(model_path, data_paths):
    # Load model
    checkpoint = torch.load(model_path, weights_only = False)
    model = StableVAE(checkpoint['input_dim'], checkpoint['window_size'])
    model.load_state_dict(checkpoint['model'])
    
    # Load data
    windows = load_data(data_paths, checkpoint['window_size'])
    dataloader, _ = prepare_dataloader(windows, 512)
    
    # Plot functions
    def plot_comparison():
        model.eval()
        batch = next(iter(dataloader))
        with torch.no_grad():
            recon, _, _ = model(batch)
        
        # Plot 3 random samples
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        for i in range(3):
            idx = np.random.randint(0, batch.size(0))
            
            # Position
            axs[i,0].plot(batch[idx,:,0].numpy(), label='Original')
            axs[i,0].plot(recon[idx,:,0].numpy(), label='Reconstructed')
            axs[i,0].set_title(f"Sample {i+1} - Position")
            
            # Velocity
            axs[i,1].plot(batch[idx,:,1].numpy(), label='Original')
            axs[i,1].plot(recon[idx,:,1].numpy(), label='Reconstructed')
            axs[i,1].set_title(f"Sample {i+1} - Velocity")
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_latent_dynamics():
        model.eval()
        latents = []
        with torch.no_grad():
            for batch in dataloader:
                mu, _ = model.encode(batch)
                latents.append(mu.numpy())
        
        latents = np.concatenate(latents)
        plt.figure(figsize=(12,6))
        plt.plot(latents[:500,:4])  # Plot first 4 dims
        plt.title("Latent Dimension Dynamics Over Time")
        plt.xlabel("Frame")
        plt.ylabel("Latent Value")
        plt.legend(["Dim 0", "Dim 1", "Dim 2", "Dim 3"])
        plt.show()
    
    plot_comparison()
    plot_latent_dynamics()

if __name__ == "__main__":
    visualize_model("vae_model.pth", ["data/walking_test_20250304_210109.csv"])