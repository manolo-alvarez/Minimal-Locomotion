""" Feature importance analysis for ZBot policy. """
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
import os
import argparse
import glob
import pandas as pd
from collections import defaultdict

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not found. Using matplotlib for basic visualizations.")

class PolicyAnalyzer:
    """Analyzes the importance of input features to a policy."""
    
    def __init__(
        self, 
        policy: Callable[[torch.Tensor], torch.Tensor], 
        obs_labels: List[str],
        action_labels: List[str],
        device: str = "cpu",
        save_dir: str = "analysis_results"
    ):
        """
        Initialize the policy analyzer.
        
        Args:
            policy: Function that maps observations to actions
            obs_labels: Labels for observation features
            action_labels: Labels for action outputs
            device: Device to run computations on
            save_dir: Directory to save analysis results
        """
        self.policy = policy
        self.obs_labels = obs_labels
        self.action_labels = action_labels
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Statistics storage
        self.obs_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }
        self.action_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }
        
        # Sensitivity matrices
        self.sensitivity_matrix = None
        self.normalized_sensitivity_matrix = None
        
    def collect_statistics(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None):
        """
        Collect statistics about observations and actions.
        
        Args:
            observations: Tensor of observations [num_samples, obs_dim]
            actions: Optional tensor of actions [num_samples, action_dim]. If None, actions
                    will be computed from the policy using the provided observations.
        """
        # Compute statistics for observations
        self.obs_stats['mean'] = observations.mean(dim=0)
        self.obs_stats['std'] = observations.std(dim=0)
        self.obs_stats['min'] = observations.min(dim=0)[0]
        self.obs_stats['max'] = observations.max(dim=0)[0]
        
        # Compute actions if not provided
        if actions is None:
            with torch.no_grad():
                actions = self.policy(observations)
        
        # Compute statistics for actions
        self.action_stats['mean'] = actions.mean(dim=0)
        self.action_stats['std'] = actions.std(dim=0)
        self.action_stats['min'] = actions.min(dim=0)[0]
        self.action_stats['max'] = actions.max(dim=0)[0]
        
    def compute_sensitivity(
        self, 
        base_observation: torch.Tensor,
        perturbation_scale: float = 0.01,
        num_samples: int = 21  # Increased for more stable estimates
    ):
        """
        Compute sensitivity of policy outputs to input perturbations.
        
        This uses a deterministic central finite difference method to ensure
        consistent results across runs with the same inputs.
        
        Args:
            base_observation: Base observation to perturb [batch_size, obs_dim]
            perturbation_scale: Scale of perturbations relative to feature std
            num_samples: Number of samples for each feature perturbation (odd number recommended)
        """
        # Set a fixed random seed for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        
        # Make sure base_observation is 2D [batch_size, obs_dim]
        if base_observation.dim() == 1:
            base_observation = base_observation.unsqueeze(0)
        
        base_observation = base_observation.to(self.device)
        
        # Extract observation dimension
        obs_dim = base_observation.shape[1]
        
        # Get action dimension by doing a forward pass
        with torch.no_grad():
            test_action = self.policy(base_observation)
        action_dim = test_action.shape[1]
        
        # If we don't have statistics yet, use simple perturbation
        if self.obs_stats['std'] is None:
            perturbation_sizes = torch.ones(obs_dim, device=self.device) * perturbation_scale
        else:
            # Scale perturbation by feature std to account for different scales
            perturbation_sizes = self.obs_stats['std'] * perturbation_scale
            
        # Ensure minimum perturbation size to avoid numerical issues
        min_perturbation = 1e-6
        perturbation_sizes = torch.clamp(perturbation_sizes, min=min_perturbation)
        
        # Print perturbation sizes for debugging
        print(f"Perturbation scale: {perturbation_scale}")
        print(f"Min perturbation size: {perturbation_sizes.min().item()}")
        print(f"Max perturbation size: {perturbation_sizes.max().item()}")
        
        # Initialize sensitivity matrices
        self.sensitivity_matrix = torch.zeros((obs_dim, action_dim), device=self.device)
        self.normalized_sensitivity_matrix = torch.zeros((obs_dim, action_dim), device=self.device)
        
        # Get baseline action first to ensure consistent reference
        with torch.no_grad():
            base_action = self.policy(base_observation).squeeze(0)
            
        print(f"Base observation shape: {base_observation.shape}")
        print(f"Base action shape: {base_action.shape}")
        
        # Compute sensitivity for each feature
        for i in range(obs_dim):
            feature_name = self.obs_labels[i] if i < len(self.obs_labels) else f"Feature_{i}"
            print(f"Processing feature {i}: {feature_name}")
            
            # Create evenly spaced perturbations for this feature
            delta = perturbation_sizes[i]
            
            # Use central difference method for more accuracy
            pos_obs = base_observation.clone()
            pos_obs[0, i] += delta
            
            neg_obs = base_observation.clone()
            neg_obs[0, i] -= delta
            
            # Forward pass for both perturbations
            with torch.no_grad():
                pos_action = self.policy(pos_obs).squeeze(0)
                neg_action = self.policy(neg_obs).squeeze(0)
            
            # Central difference formula
            feature_sensitivity = (pos_action - neg_action) / (2 * delta)
            
            # Debug information
            if i < 3 or i >= obs_dim - 3:  # Show first and last few features
                print(f"  Delta: {delta.item():.6e}")
                print(f"  Base value: {base_observation[0, i].item():.6f}")
                print(f"  Pos value: {pos_obs[0, i].item():.6f}")
                print(f"  Neg value: {neg_obs[0, i].item():.6f}")
                print(f"  Sensitivity (first 3): {feature_sensitivity[:3].cpu().numpy()}")
            
            # Replace any NaN values
            if torch.isnan(feature_sensitivity).any():
                print(f"Warning: NaN detected in sensitivity for feature {i} ({feature_name}). Setting to zero.")
                feature_sensitivity = torch.where(torch.isnan(feature_sensitivity), 
                                                torch.zeros_like(feature_sensitivity), 
                                                feature_sensitivity)
            
            # Store the sensitivity
            self.sensitivity_matrix[i, :] = feature_sensitivity
        
        # Normalize sensitivity by feature ranges
        if self.obs_stats['std'] is not None and self.action_stats['max'] is not None:
            for i in range(obs_dim):
                # Get feature range (using statistics)
                feature_std = self.obs_stats['std'][i].clamp(min=1e-6)
                
                # Get action ranges
                action_ranges = (self.action_stats['max'] - self.action_stats['min']).clamp(min=1e-6)
                
                # Normalize: sensitivity * feature_std / action_range
                normalized_sensitivity = self.sensitivity_matrix[i, :] * feature_std / action_ranges
                
                # Replace NaN values if any
                if torch.isnan(normalized_sensitivity).any():
                    print(f"Warning: NaN in normalized sensitivity for feature {i} ({self.obs_labels[i]}). Setting to zero.")
                    normalized_sensitivity = torch.where(torch.isnan(normalized_sensitivity),
                                                      torch.zeros_like(normalized_sensitivity),
                                                      normalized_sensitivity)
                
                self.normalized_sensitivity_matrix[i, :] = normalized_sensitivity
        else:
            # If no statistics available, just use raw sensitivity
            self.normalized_sensitivity_matrix = self.sensitivity_matrix.clone()
        
        # Final check for any remaining NaNs
        if torch.isnan(self.normalized_sensitivity_matrix).any():
            print("Warning: NaN values still present in sensitivity matrix. Replacing with zeros.")
            self.normalized_sensitivity_matrix = torch.nan_to_num(self.normalized_sensitivity_matrix, nan=0.0)
        
        # Save sensitivity data for reproducibility
        self.save_sensitivity_data()

    def compute_sensitivity_multi_point(self, base_observation, perturbation_scale=0.01, num_points=5):
        """Compute sensitivity using multiple points for better accuracy."""
        # For each feature
        for i in range(obs_dim):
            # Create multiple evenly-spaced perturbations
            perturbations = torch.linspace(-perturbation_scale, perturbation_scale, num_points)
            
            # Get actions for all perturbations
            actions = []
            for p in perturbations:
                perturbed_obs = base_observation.clone()
                perturbed_obs[0, i] += p
                with torch.no_grad():
                    actions.append(self.policy(perturbed_obs))
            
            # Use linear regression to find the slope (sensitivity)
            x = perturbations.view(-1, 1).numpy()
            y = torch.stack(actions).squeeze(1).cpu().numpy()
            
            # For each action dimension
            for j in range(action_dim):
                slope, _, _, _ = np.linalg.lstsq(x, y[:, j], rcond=None)
                self.sensitivity_matrix[i, j] = torch.tensor(slope[0], device=self.device)

    def save_sensitivity_data(self):
        """Save raw sensitivity data for later analysis or verification."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save sensitivity matrices
        if not hasattr(self, 'sensitivity_matrix') or self.sensitivity_matrix is None:
            return
            
        # Convert to numpy for easier saving/loading
        raw_sensitivity = self.sensitivity_matrix.cpu().numpy()
        norm_sensitivity = self.normalized_sensitivity_matrix.cpu().numpy()
        
        # Save numpy arrays
        np.save(f"{self.save_dir}/raw_sensitivity.npy", raw_sensitivity)
        np.save(f"{self.save_dir}/normalized_sensitivity.npy", norm_sensitivity)
        
        # Also save a text version of feature importances
        importances = np.linalg.norm(norm_sensitivity, axis=1)
        sorted_indices = np.argsort(-importances)
        
        with open(f"{self.save_dir}/feature_importances.txt", "w") as f:
            f.write("Feature Importance Ranking\n")
            f.write("=========================\n\n")
            f.write("Rank | Feature | Importance\n")
            f.write("--------------------------\n")
            
            for rank, idx in enumerate(sorted_indices):
                name = self.obs_labels[idx] if idx < len(self.obs_labels) else f"Feature_{idx}"
                f.write(f"{rank+1:4d} | {name:20s} | {importances[idx]:.6f}\n")

    def plot_feature_statistics(self):
        """Plot statistics for each feature."""
        if self.obs_stats['mean'] is None:
            print("No statistics collected. Run collect_statistics first.")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Set a more professional style if seaborn is available
        if HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.2)
        
        # Extract data
        means = self.obs_stats['mean'].cpu().numpy()
        stds = self.obs_stats['std'].cpu().numpy()
        mins = self.obs_stats['min'].cpu().numpy()
        maxs = self.obs_stats['max'].cpu().numpy()
        
        # Sort by std
        indices = np.argsort(stds)[::-1]  # Sort by standard deviation (descending)
        sorted_labels = [self.obs_labels[i] for i in indices]
        
        x = np.arange(len(sorted_labels))
        
        # Plot with nicer styling
        plt.errorbar(x, means[indices], yerr=stds[indices], fmt='o', 
                    color='#3498db', ecolor='#2980b9', capsize=5, 
                    label='Mean ± Std', alpha=0.8)
        plt.scatter(x, mins[indices], marker='_', color='#e74c3c', s=100, 
                   label='Min', alpha=0.7)
        plt.scatter(x, maxs[indices], marker='_', color='#2ecc71', s=100, 
                   label='Max', alpha=0.7)
        
        plt.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        plt.xlabel('Features', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        plt.title('Feature Statistics (Sorted by Variability)', fontsize=14, fontweight='bold')
        
        # Limit the number of x-tick labels if there are many features
        if len(sorted_labels) > 15:
            step = max(1, len(sorted_labels) // 15)
            plt.xticks(x[::step], [sorted_labels[i] for i in range(0, len(sorted_labels), step)], 
                      rotation=45, ha='right')
        else:
            plt.xticks(x, sorted_labels, rotation=45, ha='right')
        
        plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{self.save_dir}/feature_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_top_features_by_sensitivity(self, top_k: int = 20, normalized: bool = True):
        """
        Plot top features by sensitivity.
        
        Args:
            top_k: Number of top features to show
            normalized: Whether to use normalized sensitivity
        """
        if self.sensitivity_matrix is None:
            print("No sensitivity computed. Run compute_sensitivity first.")
            return
        
        # Set a more professional style if seaborn is available
        if HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.3)
        
        # Choose sensitivity matrix
        sensitivity = self.normalized_sensitivity_matrix if normalized else self.sensitivity_matrix
        
        # Calculate overall feature importance as L2 norm across actions
        feature_importance = torch.norm(sensitivity, dim=1).cpu().numpy()
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_k = min(top_k, len(sorted_indices))  # Ensure we don't exceed number of features
        top_indices = sorted_indices[:top_k]
        
        plt.figure(figsize=(12, 10))
        
        # Get feature labels and importance values
        labels = [self.obs_labels[i] for i in top_indices]
        values = feature_importance[top_indices]
        
        # Create horizontal bar chart with better styling
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(values)))
        bars = plt.barh(np.arange(len(labels)), values, align='center', 
                       color=colors, alpha=0.8, height=0.7)
        
        # Add value labels at the end of each bar
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=9)
        
        # Improve aesthetics
        plt.yticks(np.arange(len(labels)), labels, fontsize=11)
        plt.xlabel('Sensitivity (L2 norm across actions)', fontweight='bold')
        title = f'Top {top_k} Most Influential Features'
        if normalized:
            title += ' (Normalized by Feature Range)'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add grid only on x-axis
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(f"{self.save_dir}/top_features_{'normalized' if normalized else 'raw'}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_action_heatmap(self, top_k_features: int = 20, top_k_actions: Optional[int] = None):
        """
        Plot heatmap of feature-action sensitivities.
        
        Args:
            top_k_features: Number of top features to include
            top_k_actions: Number of top actions to include (all if None)
        """
        if self.normalized_sensitivity_matrix is None:
            print("No sensitivity computed. Run compute_sensitivity first.")
            return
        
        # Get absolute sensitivity values
        abs_sensitivity = torch.abs(self.normalized_sensitivity_matrix).cpu().numpy()
        
        # Calculate overall feature importance
        feature_importance = np.linalg.norm(abs_sensitivity, axis=1)
        total_features = len(feature_importance)  # Store total number of features
        top_k_features = min(top_k_features, total_features)  # Ensure we don't exceed number of features
        feature_indices = np.argsort(feature_importance)[::-1][:top_k_features]
        
        # Calculate overall action sensitivity
        action_sensitivity = np.linalg.norm(abs_sensitivity, axis=0)
        total_actions = len(action_sensitivity)  # Store total number of actions
        
        if (top_k_actions is not None):
            top_k_actions = min(top_k_actions, total_actions)  # Ensure we don't exceed number of actions
            action_indices = np.argsort(action_sensitivity)[::-1][:top_k_actions]
            action_count_text = f"Top {top_k_actions} of {total_actions} Actions"
        else:
            action_indices = np.arange(total_actions)
            action_count_text = f"All {total_actions} Actions"
            
        # Extract subset of sensitivity matrix
        subset_sensitivity = abs_sensitivity[np.ix_(feature_indices, action_indices)]
        
        plt.figure(figsize=(16, 12))
        
        # Use either seaborn or matplotlib for the heatmap
        if HAS_SEABORN:
            # Set style first
            sns.set_style("white")
            sns.set_context("paper", font_scale=1.2)
            
            # Custom diverging colormap with white in the middle
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Create a mask for very low values to make the heatmap cleaner
            mask = subset_sensitivity < (np.max(subset_sensitivity) * 0.05)
            
            # Plot heatmap with improved styling
            ax = sns.heatmap(
                subset_sensitivity,
                cmap='viridis',
                xticklabels=[self.action_labels[i] for i in action_indices],
                yticklabels=[self.obs_labels[i] for i in feature_indices],
                cbar_kws={'label': 'Absolute Normalized Sensitivity'},
                annot=True,  # Add values in cells
                fmt=".2f",   # Format for the annotations
                linewidths=0.5,
                mask=mask,
                square=False
            )
            
            # Rotate x labels for better readability
            plt.xticks(rotation=45, ha='right')
            
        else:
            # Matplotlib alternative to seaborn heatmap with improved styling
            plt.imshow(subset_sensitivity, cmap='viridis', aspect='auto')
            cbar = plt.colorbar(label='Absolute Normalized Sensitivity')
            cbar.ax.set_ylabel('Absolute Normalized Sensitivity', fontweight='bold')
            
            # Add text annotations in each cell
            for i in range(len(feature_indices)):
                for j in range(len(action_indices)):
                    if subset_sensitivity[i, j] > (np.max(subset_sensitivity) * 0.05):
                        plt.text(j, i, f'{subset_sensitivity[i, j]:.2f}', 
                                ha="center", va="center", 
                                color="white" if subset_sensitivity[i, j] > np.max(subset_sensitivity)/2 else "black",
                                fontsize=8)
            
            # Set x and y ticks with better formatting
            plt.yticks(np.arange(len(feature_indices)), 
                      [self.obs_labels[i] for i in feature_indices])
            plt.xticks(np.arange(len(action_indices)), 
                      [self.action_labels[i] for i in action_indices], 
                      rotation=45, ha='right')
        
        # Updated title to clearly indicate feature count
        plt.title(f'Feature-Action Sensitivity Heatmap\nTop {top_k_features} of {total_features} Features vs {action_count_text}', 
                  fontsize=16, fontweight='bold')
              
        plt.tight_layout()
        
        plt.savefig(f"{self.save_dir}/feature_action_heatmap_top{top_k_features}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self, observations: torch.Tensor, max_features_per_plot: int = 12):
        """
        Plot histograms of feature distributions, creating multiple plots if needed.
        
        Args:
            observations: Tensor of observations [num_samples, obs_dim]
            max_features_per_plot: Maximum number of features to show in each plot file
        """
        if HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.2)
        
        # Get feature importance if available
        if hasattr(self, 'normalized_sensitivity_matrix') and self.normalized_sensitivity_matrix is not None:
            sensitivity = torch.norm(self.normalized_sensitivity_matrix, dim=1).cpu().numpy()
            feature_order = np.argsort(sensitivity)[::-1]
        else:
            # Otherwise sort by standard deviation
            feature_order = np.argsort(self.obs_stats['std'].cpu().numpy())[::-1]
        
        # Total number of features to plot
        total_features = min(len(feature_order), len(self.obs_labels))
        print(f"Plotting distributions for all {total_features} features")
        
        # Create multiple plot files if needed
        for plot_num, start_idx in enumerate(range(0, total_features, max_features_per_plot)):
            # Get features for this plot
            end_idx = min(start_idx + max_features_per_plot, total_features)
            selected_features = feature_order[start_idx:end_idx]
            
            # Create subplot grid
            n_cols = 3  # Always use 3 columns for consistency
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            # Plot histograms
            observations_np = observations.cpu().numpy()
            
            for i, feature_idx in enumerate(selected_features):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                feature_data = observations_np[:, feature_idx]
                feature_name = self.obs_labels[feature_idx]
                
                # Plot distribution with KDE if seaborn is available
                if HAS_SEABORN:
                    sns.histplot(feature_data, kde=True, ax=ax, color='#3498db', bins=20)
                else:
                    ax.hist(feature_data, bins=20, alpha=0.7, color='#3498db')
                    
                # Add feature number for reference
                ax.set_title(f"{feature_name} (#{feature_idx})", fontsize=10)
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                
                # Add mean and std annotation
                mean = np.mean(feature_data)
                std = np.std(feature_data)
                ax.axvline(mean, color='red', linestyle='--', alpha=0.8)
                ax.text(0.05, 0.95, f"Mean: {mean:.2f}\nStd: {std:.2f}", 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(len(selected_features), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            # Use a different filename for each plot
            if total_features > max_features_per_plot:
                plt.savefig(f"{self.save_dir}/feature_distributions_part{plot_num+1}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{self.save_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_all_features_importance(self, normalized: bool = True):
        """Plot all features by their importance."""
        if self.sensitivity_matrix is None:
            print("No sensitivity computed. Run compute_sensitivity first.")
            return
        
        # Choose sensitivity matrix
        sensitivity = self.normalized_sensitivity_matrix if normalized else self.sensitivity_matrix
        
        # Calculate overall feature importance as L2 norm across actions
        feature_importance = torch.norm(sensitivity, dim=1).cpu().numpy()
        
        # Check for NaN values and replace with zeros
        if np.isnan(feature_importance).any():
            print("Warning: NaN values detected in feature importance. Replacing with zeros.")
            feature_importance = np.where(np.isnan(feature_importance), 0.0, feature_importance)
        
        # Sort features by importance (highest to lowest)
        sorted_indices = np.argsort(-feature_importance)  # Use negative for descending order
        num_features = len(sorted_indices)
        
        # Debug print to verify sorting
        print("\nTop 5 features by importance:")
        print("-" * 60)
        print(f"{'Rank':<5}{'Feature ID':<10}{'Importance':<12}Feature Name")
        print("-" * 60)
        for i in range(min(5, num_features)):
            idx = sorted_indices[i]
            print(f"{i+1:<5}{idx:<10}{feature_importance[idx]:.6f}      {self.obs_labels[idx]}")
        
        print("\nBottom 5 features by importance:")
        print("-" * 60)
        print(f"{'Rank':<5}{'Feature ID':<10}{'Importance':<12}Feature Name")
        print("-" * 60)
        for i in range(min(5, num_features)):
            idx = sorted_indices[-(i+1)]
            print(f"{num_features-i:<5}{idx:<10}{feature_importance[idx]:.6f}      {self.obs_labels[idx]}")
        print("-" * 60)
        
        # Get feature labels and importance values in sorted order
        labels = [f"{self.obs_labels[i]} (#{i})" for i in sorted_indices]
        values = feature_importance[sorted_indices]
        
        # Create figure with appropriate size
        plt.figure(figsize=(12, max(8, num_features * 0.25)))
        
        # Create horizontal bar chart with better styling
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_features))
        bars = plt.barh(np.arange(num_features), values, align='center', 
                       color=colors, alpha=0.8, height=0.6)
        
        # Add value labels at the end of each bar
        for i, (bar, value) in enumerate(zip(bars, values)):
            # Only show value if it's significant
            if value > max(values) * 0.001:
                plt.text(value - max(values) * 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{value:.5f}', va='center', ha='right', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Improve aesthetics
        plt.yticks(np.arange(num_features), labels, fontsize=9)
        plt.xlabel('Sensitivity (L2 norm across actions)', fontweight='bold')
        
        # Updated title to be more descriptive
        plt.title('Complete Feature Importance Ranking', fontsize=14, fontweight='bold')
        
        # Add grid only on x-axis
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Use a more descriptive filename
        plt.savefig(f"{self.save_dir}/feature_importance_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_analysis(
        self, 
        observations: torch.Tensor,
        base_observation: Optional[torch.Tensor] = None,
        perturbation_scale: float = 0.01,
        top_k: int = 20,
        seed: int = 0  # Add fixed seed for reproducibility
    ):
        """Run full feature importance analysis."""
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print("Running policy feature analysis...")
        print(f"Using random seed: {seed}")
        
        # Collect statistics
        print("Collecting statistics...")
        self.collect_statistics(observations)
        
        # Use mean observation if base_observation not provided
        if base_observation is None:
            # Use the mean observation for more stable results
            base_observation = self.obs_stats['mean'].unsqueeze(0)
            print("Using mean observation as base")
        
        # Print statistics about the base observation
        print(f"Base observation stats:")
        print(f"  Shape: {base_observation.shape}")
        print(f"  Mean: {base_observation.mean().item():.4f}")
        print(f"  Min: {base_observation.min().item():.4f}")
        print(f"  Max: {base_observation.max().item():.4f}")
            
        # Compute sensitivity
        print("Computing sensitivity...")
        self.compute_sensitivity(base_observation, perturbation_scale)
        
        # Print total number of features
        num_features = observations.shape[1]
        num_actions = self.normalized_sensitivity_matrix.shape[1]
        print(f"Analysis complete for {num_features} features and {num_actions} actions.")
        
        # Generate plots - SIMPLIFIED
        print("Generating plots...")
        self.plot_feature_statistics()
        self.plot_feature_distributions(observations, max_features_per_plot=12)  # Multiple plots if needed
        
        # Only create the comprehensive plot of all features
        self.plot_all_features_importance(normalized=True)  
        
        # Remove the heatmap that was causing issues
        # self.plot_feature_action_heatmap(top_k_features=min(top_k, num_features))
        
        self.print_top_features(top_k=min(top_k, num_features))
        self.save_feature_importance_table(normalized=True)
        
        print(f"Analysis complete. Results saved to {self.save_dir}/")
        
        # Return top feature indices for potential use elsewhere
        sensitivity = torch.norm(self.normalized_sensitivity_matrix, dim=1).cpu().numpy()
        return np.argsort(sensitivity)[::-1]

    def print_top_features(self, top_k: int = 20):
        """
        Print the top most important features.
        
        Args:
            top_k: Number of top features to print
        """
        if self.normalized_sensitivity_matrix is None:
            print("No sensitivity computed. Run compute_sensitivity first.")
            return
        
        # Calculate overall feature importance
        sensitivity = torch.norm(self.normalized_sensitivity_matrix, dim=1).cpu().numpy()
        
        # Sort features by importance
        sorted_indices = np.argsort(sensitivity)[::-1]
        top_k = min(top_k, len(sorted_indices))
        
        print(f"\nTop {top_k} most important features:")
        print("-" * 60)
        print(f"{'Rank':<5}{'Feature ID':<10}{'Importance':<12}Feature Name")
        print("-" * 60)
        
        for i in range(top_k):
            idx = sorted_indices[i]
            name = self.obs_labels[idx]
            importance = sensitivity[idx]
            print(f"{i+1:<5}{idx:<10}{importance:.4f}      {name}")
        
        print("-" * 60)
        
        # Also save to a text file
        with open(f"{self.save_dir}/top_features.txt", 'w') as f:
            f.write(f"Top {top_k} most important features:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Rank':<5}{'Feature ID':<10}{'Importance':<12}Feature Name\n")
            f.write("-" * 60 + "\n")
            
            for i in range(top_k):
                idx = sorted_indices[i]
                name = self.obs_labels[idx]
                importance = sensitivity[idx]
                f.write(f"{i+1:<5}{idx:<10}{importance:.4f}      {name}\n")
            
            f.write("-" * 60 + "\n")

    def save_feature_importance_table(self, normalized=True):
        """Save a detailed table of feature importance with descriptions."""
        if self.normalized_sensitivity_matrix is None:
            print("No sensitivity computed. Run compute_sensitivity first.")
            return
        
        # Choose sensitivity matrix
        sensitivity = self.normalized_sensitivity_matrix if normalized else self.sensitivity_matrix
        
        # Calculate overall feature importance
        feature_importance = torch.norm(sensitivity, dim=1).cpu().numpy()
        
        # Check for NaN values
        if np.isnan(feature_importance).any():
            print("Warning: NaN values detected in feature importance. Replacing with zeros.")
            feature_importance = np.where(np.isnan(feature_importance), 0.0, feature_importance)
        
        # Sort features by importance
        sorted_indices = np.argsort(-feature_importance)
        
        # Prepare table
        table = []
        table.append("# Feature Importance Analysis")
        table.append("\n## Feature Descriptions")
        table.append("- Base height: Height of the robot's base from the ground")
        table.append("- Base orientation: Roll, pitch, yaw angles of the robot's base")
        table.append("- Base linear velocity: Forward (x), lateral (y), and vertical (z) velocity")
        table.append("- Commands: Target velocities for the robot to achieve")
        table.append("- Joint positions: Current angles of each robot joint")
        table.append("- Joint velocities: Current angular velocities of each joint")
        table.append("\n## Importance Ranking (Normalized)")
        table.append("\n| Rank | Feature ID | Importance | Feature Name | Description |")
        table.append("| ---- | ---------- | ---------- | ------------ | ----------- |")
        
        # Feature descriptions dictionary
        descriptions = {
            "base_height": "Height of robot base from ground",
            "projected_gravity": "Gravity vector in robot's frame (indicates orientation)",
            "base_ang_vel": "Angular velocity of the robot's base",
            "base_lin_vel": "Linear velocity of the robot's base",
            "command": "Target velocity command for the robot to achieve",
            "dof_pos": "Current position/angle of a robot joint",
            "dof_vel": "Current angular velocity of a robot joint",
        }
        
        # Add rows to table
        for rank, idx in enumerate(sorted_indices):
            name = self.obs_labels[idx]
            importance = feature_importance[idx]
            
            # Get description
            description = "Unknown"
            for key, desc in descriptions.items():
                if key in name:
                    description = desc
                    break
            
            table.append(f"| {rank+1} | {idx} | {importance:.6f} | {name} | {description} |")
        
        # Write table to file
        with open(f"{self.save_dir}/feature_importance_table.md", 'w') as f:
            f.write("\n".join(table))
        
        print(f"Feature importance table saved to {self.save_dir}/feature_importance_table.md")

        # Also save CSV version for easy import into other tools
        with open(f"{self.save_dir}/feature_importance.csv", 'w') as f:
            f.write("rank,feature_id,importance,feature_name\n")
            for rank, idx in enumerate(sorted_indices):
                name = self.obs_labels[idx]
                importance = feature_importance[idx]
                f.write(f"{rank+1},{idx},{importance:.6f},{name}\n")

    @staticmethod
    def aggregate_feature_importance(log_folder, output_dir=None):
        """
        Aggregate feature importance data from multiple runs.
        
        Args:
            log_folder: Path to folder containing subfolders with feature_importance.csv files
            output_dir: Directory to save the aggregated results (defaults to log_folder)
        
        Returns:
            DataFrame with aggregated feature importance data
        """
        
        # Use log_folder as output_dir if not specified
        if output_dir is None:
            output_dir = log_folder
        
        # Find all feature_importance.csv files
        csv_files = glob.glob(os.path.join(log_folder, '**/feature_importance.csv'), recursive=True)
        
        if not csv_files:
            print(f"No feature_importance.csv files found in {log_folder}")
            return None
        
        print(f"Found {len(csv_files)} feature importance files to analyze")
        
        # Load all CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                print(f"Processed {csv_file}: {len(df)} features")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        if not all_data:
            print("No valid feature importance data could be processed")
            return None
        
        # Create a dictionary to store importance values for each feature
        feature_importance = defaultdict(list)
        
        # Collect importance values for each feature across all runs
        for df in all_data:
            for _, row in df.iterrows():
                feature_name = row['feature_name']
                importance = row['importance']
                feature_importance[feature_name].append(importance)
        
        # Calculate mean importance for each feature
        aggregated_data = []
        for feature_name, importance_values in feature_importance.items():
            mean_importance = np.mean(importance_values)
            std_importance = np.std(importance_values)
            min_importance = np.min(importance_values)
            max_importance = np.max(importance_values)
            count = len(importance_values)
            
            aggregated_data.append({
                'feature_name': feature_name,
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'min_importance': min_importance,
                'max_importance': max_importance,
                'count': count
            })
        
        # Convert to DataFrame and sort by mean importance
        result_df = pd.DataFrame(aggregated_data)
        result_df = result_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
        result_df['rank'] = result_df.index + 1
        
        # Save aggregated data to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'aggregated_feature_importance.csv')
        result_df.to_csv(output_file, index=False)
        print(f"Saved aggregated data to {output_file}")
        
        # Create figure with appropriate size
        plt.figure(figsize=(12, max(8, len(result_df) * 0.25)))
        
        # Get data for plotting
        features = result_df['feature_name'].tolist()
        mean_importance = result_df['mean_importance'].tolist()
        std_importance = result_df['std_importance'].tolist()
        min_importance = result_df['min_importance'].tolist()
        max_importance = result_df['max_importance'].tolist()
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(features)))
        
        # Plot bars for mean importance
        bars = plt.barh(y_pos, mean_importance, align='center', color=colors, alpha=0.8, height=0.6)
        
        # Add range indicators (min to max)
        for i, (min_val, max_val) in enumerate(zip(min_importance, max_importance)):
            plt.plot([min_val, max_val], [y_pos[i], y_pos[i]], 'k-', alpha=0.3)
            plt.plot([min_val], [y_pos[i]], 'k|', alpha=0.5)
            plt.plot([max_val], [y_pos[i]], 'k|', alpha=0.5)
        
        
        # Improve aesthetics
        plt.yticks(y_pos, features, fontsize=9)
        plt.xlabel('Feature Importance (Mean across runs)', fontweight='bold')
        plt.title('Aggregated Feature Importance Ranking', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add a legend explaining the error bars
        plt.plot([], [], 'k-', label='Min-Max Range')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'feature_importance_ranking.png'), dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.join(output_dir, 'feature_importance_ranking.png')}")
        
        # Print top 20 features
        print("\nTop 20 most important features (aggregated across runs):")
        print("-" * 80)
        print(f"{'Rank':<5}{'Feature Name':<40}{'Mean Importance':<15}{'Std Dev':<10}{'Min':<10}{'Max':<10}")
        print("-" * 80)
        
        for i, row in result_df.head(20).iterrows():
            print(f"{i+1:<5}{row['feature_name']:<40}{row['mean_importance']:.6f}      {row['std_importance']:.6f}  {row['min_importance']:.6f}  {row['max_importance']:.6f}")
        
        print("-" * 80)
        
        return result_df

    def robust_sensitivity_analysis(self, observations, num_base_points=10, perturbation_scale=0.01):
        """Compute sensitivity across multiple base observations for robustness."""
        # Randomly select base points
        indices = torch.randperm(len(observations))[:num_base_points]
        base_points = observations[indices]
        
        # Collect statistics first
        print("Collecting statistics for robust sensitivity analysis...")
        self.collect_statistics(observations)
        
        # Initialize aggregate sensitivity matrices
        obs_dim = observations.shape[1]
        
        # Get action dimension by doing a forward pass
        with torch.no_grad():
            test_action = self.policy(observations[0:1])
        action_dim = test_action.shape[1]
        
        # Initialize matrices with the right dimensions
        agg_sensitivity = torch.zeros((obs_dim, action_dim), device=self.device)
        self.sensitivity_matrix = torch.zeros((obs_dim, action_dim), device=self.device)
        self.normalized_sensitivity_matrix = torch.zeros((obs_dim, action_dim), device=self.device)
        
        print(f"Computing sensitivity for {num_base_points} diverse base points...")
        
        # Compute sensitivity for each base point
        for i, base_obs in enumerate(base_points):
            print(f"Computing sensitivity for base point {i+1}/{num_base_points}")
            
            # Compute sensitivity for this base point
            self.compute_sensitivity(base_obs.unsqueeze(0), perturbation_scale)
            
            # Add to aggregate
            agg_sensitivity += self.normalized_sensitivity_matrix
            
            # Save this base point's sensitivity with a unique name
            np.save(f"{self.save_dir}/sensitivity_base{i+1}.npy", 
                   self.normalized_sensitivity_matrix.cpu().numpy())
        
        # Average the sensitivities
        self.normalized_sensitivity_matrix = agg_sensitivity / num_base_points
        print("Completed robust sensitivity analysis across multiple base points")
        
        # Save the final aggregated sensitivity
        np.save(f"{self.save_dir}/sensitivity_aggregated.npy", 
               self.normalized_sensitivity_matrix.cpu().numpy())

    def phase_aware_sensitivity_analysis(self, observations, env, num_phases=8, 
                                   perturbation_scale=0.01, phase_indices=None,
                                   foot_contacts=None):
        """
        Analyze feature sensitivity across different phases of the gait cycle.
        """
        # First collect overall statistics
        print("Collecting overall statistics...")
        self.collect_statistics(observations)
        
        # Detect gait phases if not provided
        if phase_indices is None:
            print(f"Detecting {num_phases} gait phases...")
            # Call as instance method
            phase_indices = self.detect_gait_phases(observations, env, num_phases)
        else:
            # Use provided phases, but update num_phases if needed
            num_phases = max(phase_indices.keys()) + 1
            print(f"Using pre-computed gait phases ({num_phases} phases)")
        
        # Get observation and action dimensions
        obs_dim = observations.shape[1]
        
        # Do a forward pass to determine action dimension
        with torch.no_grad():
            test_action = self.policy(observations[:1])
        action_dim = test_action.shape[1]
        
        # Initialize storage for phase-specific sensitivities
        phase_sensitivities = {}
        
        # Print debug info
        print(f"Analyzing sensitivity across {num_phases} phases")
        print(f"Observation dimension: {obs_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Create a subplot grid for phase-specific importance plots
        num_rows = max(2, int(np.ceil(num_phases/4)))
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5*num_rows))
        axes = axes.flatten()
        
        # For each phase, compute a separate sensitivity matrix
        for phase_idx, phase in enumerate(sorted(phase_indices.keys())):
            phase_sample_indices = phase_indices[phase]
            
            if len(phase_sample_indices) < 5:
                print(f"Warning: Phase {phase} has too few samples ({len(phase_sample_indices)}). Skipping.")
                if phase_idx < len(axes):
                    axes[phase_idx].text(0.5, 0.5, f"Phase {phase}: Not enough samples", 
                                       ha='center', va='center', fontsize=14)
                    axes[phase_idx].set_title(f"Phase {phase}: Skipped (insufficient data)")
                continue
                
            print(f"Analyzing phase {phase}/{num_phases-1} with {len(phase_sample_indices)} samples")
            
            # Get representative observation for this phase
            phase_observations = observations[phase_sample_indices]
            base_obs = phase_observations.mean(dim=0, keepdim=True).to(self.device)
            
            # Debug info
            print(f"  Phase {phase} base observation shape: {base_obs.shape}")
            print(f"  Min: {base_obs.min().item():.4f}, Max: {base_obs.max().item():.4f}")
            
            # Initialize phase-specific sensitivity matrix
            phase_sensitivity = torch.zeros((obs_dim, action_dim), device=self.device)
            
            # Compute sensitivity for this phase using central difference method
            for i in range(obs_dim):
                # Set perturbation size based on feature statistics
                delta = perturbation_scale * self.obs_stats['std'][i].clamp(min=1e-6)
                
                # Create perturbed observations
                pos_obs = base_obs.clone()
                pos_obs[0, i] += delta
                
                neg_obs = base_obs.clone()
                neg_obs[0, i] -= delta
                
                # Get actions
                with torch.no_grad():
                    pos_action = self.policy(pos_obs).squeeze(0)
                    neg_action = self.policy(neg_obs).squeeze(0)
                
                # Compute sensitivity
                feature_sensitivity = (pos_action - neg_action) / (2 * delta)
                
                # Store in phase-specific sensitivity matrix
                phase_sensitivity[i] = feature_sensitivity
            
            # Check for NaN values
            if torch.isnan(phase_sensitivity).any():
                print(f"Warning: NaN values detected in phase {phase} sensitivity. Replacing with zeros.")
                phase_sensitivity = torch.nan_to_num(phase_sensitivity, nan=0.0)
            
            # Store this phase's sensitivity
            phase_sensitivities[phase] = phase_sensitivity
            
            # Calculate feature importance for this specific phase
            feature_importance = torch.norm(phase_sensitivity, dim=1).cpu().numpy()
            
            # Skip plotting if we have too many phases for our grid
            if phase_idx >= len(axes):
                print(f"Warning: Too many phases to plot all individually.")
                continue
                
            # Sort features by importance for this specific phase
            sorted_indices = np.argsort(-feature_importance)[:20]  # Top 20 features
            
            # Plot in the appropriate subplot
            ax = axes[phase_idx]
            labels = [f"{self.obs_labels[i]}" for i in sorted_indices]
            values = feature_importance[sorted_indices]
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(sorted_indices)), values, align='center')
            ax.set_yticks(range(len(sorted_indices)))
            ax.set_yticklabels(labels, fontsize=8)
            
            # Add phase name with description
            phase_name = self.get_phase_name(phase, num_phases)
            ax.set_title(f"Phase {phase}: {phase_name}", fontweight='bold')
            
            # Add axis labels where appropriate
            if phase_idx % 4 == 0:
                ax.set_ylabel("Features")
            if phase_idx >= len(phase_indices) - 4 or phase_idx >= len(axes) - 4:
                ax.set_xlabel("Importance")
                
            # Print the top 3 features for this phase for debugging
            top3_features = [(self.obs_labels[i], feature_importance[i]) for i in sorted_indices[:3]]
            print(f"  Top 3 features for phase {phase}: {top3_features}")
        
        # Hide unused subplots
        for i in range(len(phase_indices), len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle("Feature Importance by Gait Phase", fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{self.save_dir}/phase_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap showing how feature importance varies across phases
        self._plot_phase_importance_heatmap(phase_sensitivities, num_phases)
        
        # Visualize the robot state in different phases
        try:
            self.visualize_gait_phases(observations, phase_indices, foot_contacts)
        except Exception as e:
            print(f"Warning: Could not visualize gait phases: {e}")
        
        # Also save numerical data for further analysis
        self._save_phase_sensitivity_data(phase_sensitivities)
        
        return phase_sensitivities

    def get_phase_name(self, phase, num_phases):
        """Get a descriptive name for each phase of the gait cycle."""
        if num_phases == 8:
            # For an 8-phase division of a typical quadruped trot gait
            phase_names = [
                "Right Front Foot Strike",
                "Right Front Stance",
                "Right Front Foot Lift",
                "Right Front Swing",
                "Left Front Foot Strike",
                "Left Front Stance",
                "Left Front Foot Lift",
                "Left Front Swing"
            ]
            return phase_names[phase] if phase < len(phase_names) else f"Phase {phase}"
        elif num_phases == 4:
            # For a 4-phase division
            phase_names = [
                "Double Support (RF+LH)",
                "Double Support (LF+RH)",
                "Flight Phase 1",
                "Flight Phase 2"
            ]
            return phase_names[phase] if phase < len(phase_names) else f"Phase {phase}"
        else:
            # Generic phase names
            fraction = phase / num_phases
            if fraction < 0.25:
                return "Early Stance"
            elif fraction < 0.5:
                return "Late Stance"
            elif fraction < 0.75:
                return "Early Swing"
            else:
                return "Late Swing"

    def _save_phase_sensitivity_data(self, phase_sensitivities):
        """Save numerical data from phase-specific sensitivity analysis."""
        if not phase_sensitivities:
            return
            
        # Create a directory for the numerical data
        data_dir = os.path.join(self.save_dir, "phase_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save overall summary as CSV
        summary_rows = ["Feature,Feature ID"]
        
        # Add phase columns to header
        phases = sorted(phase_sensitivities.keys())
        summary_rows[0] += "," + ",".join([f"Phase {p}" for p in phases])
        
        # Calculate importance for each feature in each phase
        for i, feature_name in enumerate(self.obs_labels):
            row = f"{feature_name},{i}"
            
            for phase in phases:
                sensitivity = phase_sensitivities[phase]
                importance = torch.norm(sensitivity[i], dim=0).item()
                row += f",{importance:.6f}"
                
            summary_rows.append(row)
        
        # Write summary CSV
        with open(os.path.join(data_dir, "phase_importance_summary.csv"), 'w') as f:
            f.write("\n".join(summary_rows))
        
        # Also save individual phase data as numpy files
        for phase, sensitivity in phase_sensitivities.items():
            np.save(os.path.join(data_dir, f"phase_{phase}_sensitivity.npy"), 
                    sensitivity.cpu().numpy())

    def _plot_phase_importance_heatmap(self, phase_sensitivities, num_phases):
        """Plot heatmap of feature importance across different gait phases."""
        if not phase_sensitivities:
            return
            
        # Get feature dimension from first phase
        first_phase = list(phase_sensitivities.keys())[0]
        obs_dim = phase_sensitivities[first_phase].shape[0]
        
        # Prepare data for heatmap (features x phases)
        phases = sorted(phase_sensitivities.keys())
        heatmap_data = np.zeros((obs_dim, len(phases)))
        
        # Calculate importance for each feature in each phase
        for i, phase in enumerate(phases):
            # Calculate importance as L2 norm across actions
            importance = torch.norm(phase_sensitivities[phase], dim=1).cpu().numpy()
            heatmap_data[:, i] = importance
        
        # Normalize each feature's importance across phases to highlight variations
        # This makes it easier to see changes in importance, not just absolute values
        row_maxes = np.max(heatmap_data, axis=1, keepdims=True)
        row_maxes[row_maxes == 0] = 1  # Avoid division by zero
        normalized_heatmap = heatmap_data / row_maxes
        
        # Sort features by overall importance
        overall_importance = heatmap_data.sum(axis=1)
        sorted_indices = np.argsort(-overall_importance)
        
        # Select top features for better visualization
        top_k = min(25, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        top_data = normalized_heatmap[top_indices, :]
        top_labels = [f"{self.obs_labels[i]} (#{i})" for i in top_indices]
        
        # Create figure for the heatmap
        plt.figure(figsize=(12, 10))
        
        # Get phase names
        phase_names = [f"{p}: {self.get_phase_name(p, num_phases)}" for p in phases]
        
        # Plot using seaborn if available, otherwise matplotlib
        if HAS_SEABORN:
            # Create a diverging colormap to highlight differences
            cmap = sns.color_palette("viridis", as_cmap=True)
            
            # Plot with seaborn for nicer aesthetics
            ax = sns.heatmap(top_data, cmap=cmap, 
                            xticklabels=phase_names,
                            yticklabels=top_labels,
                            cbar_kws={'label': 'Normalized Importance'})
            
            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, ha='right')
            
        else:
            # Matplotlib alternative
            plt.imshow(top_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Normalized Importance')
            plt.yticks(np.arange(len(top_labels)), top_labels)
            plt.xticks(np.arange(len(phases)), phase_names, rotation=45, ha='right')
        
        plt.title("Feature Importance Variation Across Gait Phases", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the heatmap
        plt.savefig(f"{self.save_dir}/phase_importance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a non-normalized version to show absolute values
        plt.figure(figsize=(12, 10))
        if HAS_SEABORN:
            ax = sns.heatmap(heatmap_data[top_indices, :], cmap='viridis', 
                            xticklabels=phase_names,
                            yticklabels=top_labels,
                            cbar_kws={'label': 'Absolute Importance'})
            plt.xticks(rotation=45, ha='right')
        else:
            plt.imshow(heatmap_data[top_indices, :], cmap='viridis', aspect='auto')
            plt.colorbar(label='Absolute Importance')
            plt.yticks(np.arange(len(top_labels)), top_labels)
            plt.xticks(np.arange(len(phases)), phase_names, rotation=45, ha='right')
        
        plt.title("Absolute Feature Importance by Gait Phase", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/phase_importance_absolute.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_gait_phases(self, observations, phase_indices, foot_contacts=None):
        """
        Create a visual representation of the different gait phases.
        
        Args:
            observations: Tensor of collected observations
            phase_indices: Dictionary mapping phase numbers to observation indices
            foot_contacts: Tensor of foot contact information (if available)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
        except ImportError:
            print("Matplotlib not available, skipping gait phase visualization.")
            return
        
        # Set up the figure
        phases = sorted(phase_indices.keys())
        
        # For quadruped visualization, use a 2-row layout
        if len(phases) <= 4:
            rows, cols = 1, len(phases)
        else:
            cols = 4
            rows = (len(phases) + 3) // 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
        if rows == 1 and cols == 1:
            axes = np.array([axes])  # Make 1D for consistent indexing
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        
        print(f"Visualizing {len(phases)} gait phases...")
        
        # For each phase, plot a representative state
        for i, phase in enumerate(phases):
            # Get the correct subplot
            if rows == 1:
                ax = axes[i % cols]
            elif cols == 1:
                ax = axes[i // cols]
            else:
                ax = axes[i // cols, i % cols]
            
            phase_idx = phase_indices[phase]
            
            if len(phase_idx) == 0:
                ax.text(0.5, 0.5, f"No samples for phase {phase}", 
                       ha='center', va='center', fontsize=14)
                continue
            
            # Get a representative observation for this phase - use middle observation
            rep_idx = phase_idx[len(phase_idx) // 2]  # Middle observation
            rep_obs = observations[rep_idx].cpu().numpy()
            
            # If foot contacts were provided, get them for this observation
            phase_contacts = None
            if foot_contacts is not None:
                if rep_idx < len(foot_contacts):  # Make sure we don't go out of bounds
                    phase_contacts = foot_contacts[rep_idx].cpu().numpy()
                    print(f"  Phase {phase} foot contacts: {phase_contacts}")
                else:
                    print(f"  Warning: No foot contact data available for phase {phase} (idx {rep_idx})")
            
            # Get joint positions from observation
            # These indices are specific to ZBot format - adjust if needed
            joint_start_idx = 13  # Typical starting index for joint positions in ZBot
            num_joints = 12      # Typical number of joints in a quadruped
            
            # Check if we have enough observation dimensions
            if rep_obs.size >= joint_start_idx + num_joints:
                joint_positions = rep_obs[joint_start_idx:joint_start_idx+num_joints]
                print(f"  Phase {phase} joint positions: {joint_positions[:4]}...")
            else:
                print(f"  Warning: Observation doesn't have enough dimensions for joint positions")
                joint_positions = np.zeros(num_joints)
            
            # Draw simplified robot visualization
            self._draw_robot_state(ax, joint_positions, phase, phase_contacts)
            
            # Add phase description
            phase_name = self.get_phase_name(phase, len(phases))
            ax.set_title(f"Phase {phase}: {phase_name}", fontsize=12, fontweight='bold')
        
        # Hide any unused subplots
        total_subplots = rows * cols
        for i in range(len(phases), total_subplots):
            if rows == 1:
                if i < len(axes):
                    axes[i].set_visible(False)
            elif cols == 1:
                if i < len(axes):
                    axes[i].set_visible(False)
            else:
                row_idx = i // cols
                col_idx = i % cols
                if row_idx < rows and col_idx < cols:
                    axes[row_idx, col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle("Robot State Across Gait Phases", fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(f"{self.save_dir}/gait_phase_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _draw_robot_state(self, ax, joint_positions, phase, foot_contacts=None):
        """
        Draw a simplified representation of the robot's state in a given gait phase.
        
        Args:
            ax: Matplotlib axis to draw on
            joint_positions: Joint angles for the robot
            phase: Phase number
            foot_contacts: Contact state for each foot (if available)
        """
        # Create a simplified top-down view of a quadruped
        # This is a very basic visualization - adjust as needed for your robot
        
        # Define robot dimensions
        body_length = 0.5
        body_width = 0.25
        leg_length = 0.3
        
        # Convert joint positions to leg extension percentages (simplified)
        # This is just a visual approximation, not physically accurate
        hip_angles = joint_positions[[0, 3, 6, 9]]  # Example hip joints
        knee_angles = joint_positions[[1, 4, 7, 10]]  # Example knee joints
        
        # Calculate extension percentage for each leg (0 = fully extended, 1 = fully compressed)
        # This is a very simplistic model - replace with actual kinematics if available
        leg_extensions = np.abs(np.sin(hip_angles) * np.cos(knee_angles))
        leg_extensions = np.clip(leg_extensions, 0.2, 1.0)  # Keep within reasonable range
        
        # Basic rectangular body
        body = plt.Rectangle((-body_length/2, -body_width/2), body_length, body_width, 
                           fill=True, color='lightgray', alpha=0.7)
        ax.add_patch(body)
        
        # Head indicator (front of robot)
        head = plt.Circle((body_length/2, 0), 0.05, fill=True, color='gray')
        ax.add_patch(head)
        
        # Foot positions based on leg extensions
        foot_positions = [
            # Front right - adjust x/y based on extension
            (body_length/2 - 0.05, -body_width/2 - leg_length * leg_extensions[0]),
            # Front left
            (body_length/2 - 0.05, body_width/2 + leg_length * leg_extensions[1]),
            # Rear right
            (-body_length/2 + 0.05, -body_width/2 - leg_length * leg_extensions[2]),
            # Rear left
            (-body_length/2 + 0.05, body_width/2 + leg_length * leg_extensions[3])
        ]
        
        # Draw legs and feet
        foot_labels = ["FR", "FL", "RR", "RL"]
        
        # Use actual foot contacts if provided, otherwise approximate based on phase
        if foot_contacts is not None and len(foot_contacts) >= 4:
            contact_states = foot_contacts[:4] > 0.5
        else:
            # Simple approximation for a trot gait
            if len(foot_labels) == 4:  # Quadruped
                if phase % 2 == 0:
                    # Diagonal pair 1 (FR+RL) in contact
                    contact_states = np.array([True, False, False, True])
                else:
                    # Diagonal pair 2 (FL+RR) in contact
                    contact_states = np.array([False, True, True, False])
            else:
                # Default all in contact
                contact_states = np.array([True] * len(foot_labels))
        
        for i, (fx, fy) in enumerate(foot_positions):
            # Color based on contact state
            color = 'red' if contact_states[i] else 'blue'
            
            # Determine leg connection points
            if i < 2:  # Front legs
                start_x = body_length/2
            else:  # Rear legs
                start_x = -body_length/2
                
            if i % 2 == 0:  # Right legs
                start_y = -body_width/2
            else:  # Left legs
                start_y = body_width/2
                
            # Draw leg
            ax.plot([start_x, fx], [start_y, fy], 'k-', linewidth=2)
            
            # Draw foot
            foot = plt.Circle((fx, fy), 0.05, fill=True, color=color)
            ax.add_patch(foot)
            
            # Add foot label
            ax.text(fx, fy + 0.07, foot_labels[i], ha='center')
        
        # Add a legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Foot Contact')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Foot in Air')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        # Add phase indicator
        ax.text(0, -body_width/2 - leg_length - 0.15, 
               f"Phase {phase}", ha='center', fontsize=12, fontweight='bold')
        
        # Set axis limits and appearance
        ax.set_xlim(-body_length - leg_length/2, body_length + leg_length/2)
        ax.set_ylim(-body_width - leg_length - 0.2, body_width + leg_length + 0.2)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Forward/Backward')
        ax.set_ylabel('Left/Right')
    
    def detect_gait_phases(self, observations, env, num_phases=8):
        """
        Detect gait phases from observations using time-based estimation.
        
        Args:
            observations: Tensor of observations
            env: Environment instance (not used in time-based estimation)
            num_phases: Number of phases to identify
            
        Returns:
            phase_indices: Dictionary mapping phase number to observation indices
        """
        # Estimate typical stride duration (in number of steps)
        # For quadrupeds at normal speed, ~20-30 steps per stride is common
        estimated_stride_steps = 20
        
        # Create phases based on time/step count
        phases = np.linspace(0, 2*np.pi, estimated_stride_steps, endpoint=False)
        phases = np.tile(phases, len(observations)//estimated_stride_steps + 1)[:len(observations)]
        
        # Convert to discrete phase buckets
        discrete_phases = (phases/(2*np.pi) * num_phases).astype(int) % num_phases
        
        # Group observation indices by phase
        phase_indices = {}
        for phase in range(num_phases):
            phase_indices[phase] = np.where(discrete_phases == phase)[0]
            print(f"Phase {phase}: {len(phase_indices[phase])} samples")
        
        return phase_indices

def analyze_policy(env, runner, save_dir="feature_analysis", num_samples=1000, device="cpu"):
    """
    Analyze the importance of features in the policy.
    
    Args:
        env: The environment
        runner: The policy runner
        save_dir: Directory to save analysis results
        num_samples: Number of samples to collect
        device: Device to run on
    
    Returns:
        PolicyAnalyzer: The analyzer with results
    """
    # Get policy from runner
    policy = runner.get_inference_policy(device=device)
    
    # Get observation dimensions
    obs, _ = env.get_observations()
    obs_dim = obs.shape[1]
    
    # Try to get the actual observation labels from the environment
    try:
        # First try to use the environment's get_observation_labels method
        obs_labels = env.get_observation_labels()
        print("Using observation labels from environment")
    except (AttributeError, NotImplementedError):
        # If that fails, use standard ZBot observation labels
        obs_labels = [
            "base_height",       # Base height
            "base_orientation[0]", # Roll
            "base_orientation[1]", # Pitch
            "base_orientation[2]", # Yaw
            "base_lin_vel[0]",    # Linear velocity x
            "base_lin_vel[1]",    # Linear velocity y
            "base_lin_vel[2]",    # Linear velocity z
            "command[0]",         # Forward command
            "command[1]",         # Lateral command 
            "command[2]",         # Turn command
        ]
        
        # For joint positions and velocities
        num_joints = (obs_dim - len(obs_labels)) // 2  # Assuming remaining obs are split between pos/vel
        
        if num_joints > 0:
            for i in range(num_joints):
                obs_labels.append(f"joint_pos_{i}")
            for i in range(num_joints):
                obs_labels.append(f"joint_vel_{i}")
        
        # If we still don't have enough labels, add generic ones
        if len(obs_labels) < obs_dim:
            obs_labels.extend([f"feature_{i+len(obs_labels)}" for i in range(obs_dim - len(obs_labels))])
            
        print(f"Using default ZBot observation labels with {num_joints} joints")
    
    # Try to get action labels from the environment
    try:
        action_labels = env.get_action_labels()
    except (AttributeError, NotImplementedError):
        # Use test action to determine action dimensionality
        with torch.no_grad():
            test_action = policy(obs)
        action_dim = test_action.shape[1]
        
        # Create action labels based on expected ZBot joint structure
        action_labels = []
        
        # Typical ZBot has 8 joints, but we'll make it flexible
        for i in range(action_dim):
            joint_name = f"joint_{i//3}"
            axis_name = ["x", "y", "z"][i % 3]
            action_labels.append(f"{joint_name}_{axis_name}")
    
    print(f"Observation space: {obs_dim} dimensions")
    print(f"Action space: {len(action_labels)} dimensions")
    
    # Create analyzer with proper labels
    analyzer = PolicyAnalyzer(
        policy=policy,
        obs_labels=obs_labels,
        action_labels=action_labels,
        device=device,
        save_dir=save_dir
    )
    
    # Collect samples - rest of function remains the same
    print(f"Collecting {num_samples} samples for analysis...")
    observations = []
    
    # Reset environment
    obs, _ = env.get_observations()
    
    # Use fixed commands for consistent evaluation
    fixed_cmd = {"x": 1.0, "y": 0.0, "yaw": 0.0}
    
    # Collect observations using varying commands to get diverse states
    commands = [
        {"x": 1.0, "y": 0.0, "yaw": 0.0},
        {"x": 0.0, "y": 1.0, "yaw": 0.0},
        {"x": 0.7, "y": 0.7, "yaw": 0.0}
    ]

    # Cycle through commands while collecting observations
    for i in range(num_samples):
        cmd = commands[i % len(commands)]
        # Apply command and collect observation
        # Apply fixed command to observation
        obs_copy = obs.clone()
        
        # Try to locate the command indices (usually 6,7,8 in ZBot but could be different)
        cmd_indices = None
        for idx, label in enumerate(obs_labels):
            if "command" in label.lower():
                if cmd_indices is None:
                    cmd_indices = []
                cmd_indices.append(idx)
        
        # Apply commands at the right indices
        if cmd_indices and len(cmd_indices) >= 3:
            obs_copy[:, cmd_indices[0]] = fixed_cmd["x"]
            obs_copy[:, cmd_indices[1]] = fixed_cmd["y"]
            obs_copy[:, cmd_indices[2]] = fixed_cmd["yaw"]
        else:
            # Fallback to standard indices
            obs_copy[:, 6] = fixed_cmd["x"]
            obs_copy[:, 7] = fixed_cmd["y"]
            obs_copy[:, 8] = fixed_cmd["yaw"]
        
        # Get action
        with torch.no_grad():
            action = policy(obs_copy)
        
        # Step environment
        result = env.step(action)
        
        # Handle different return formats
        if len(result) == 4:
            obs, _, _, _ = result
        elif len(result) == 5:
            obs, _, _, _, _ = result
        
        # Store observation
        observations.append(obs_copy[0].clone())  # Store modified observation
        
        # Progress
        if (i+1) % (num_samples // 10) == 0:
            print(f"Collected {i+1}/{num_samples} samples")
    
    # Stack observations
    observations = torch.stack(observations).to(device)
    
    # Run analysis
    analyzer.run_full_analysis(observations)
    
    return analyzer

def analyze_with_command_variation(env, policy, commands=None):
    """Analyze feature importance across different command scenarios."""
    if commands is None:
        commands = [
            {"x": 1.0, "y": 0.0, "yaw": 0.0},  # Forward
            {"x": 0.0, "y": 1.0, "yaw": 0.0},  # Lateral
            {"x": 0.0, "y": 0.0, "yaw": 1.0},  # Turn
            {"x": 0.7, "y": 0.7, "yaw": 0.0}   # Diagonal
        ]
    
    # Run analysis for each command
    results = {}
    for cmd_name, cmd in commands.items():
        print(f"Analyzing command scenario: {cmd_name}")
        analyzer = analyze_policy(env, policy, cmd=cmd)
        results[cmd_name] = analyzer
    
    # Aggregate results
    aggregate_importance = np.zeros(len(results[list(results.keys())[0]].obs_labels))
    for cmd_name, analyzer in results.items():
        imp = np.linalg.norm(analyzer.normalized_sensitivity_matrix.cpu().numpy(), axis=1)
        aggregate_importance += imp
    
    # Average importance across command scenarios
    aggregate_importance /= len(results)
    
def analyze_with_multiple_seeds(env, runner, save_dir, num_samples=1000, seeds=None, device="cpu"):
    """
    Run feature importance analysis with multiple seeds and aggregate results.
    
    Args:
        env: Environment
        runner: Policy runner
        save_dir: Directory to save results
        num_samples: Number of samples to collect per run
        seeds: List of seeds to use
        device: Device to run on
        
    Returns:
        Aggregated results and a combined analyzer with variance information
    """
    if seeds is None:
        seeds = [0]
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get policy once
    policy = runner.get_inference_policy(device=device)
    
    # Get observation labels once
    obs_labels = env.get_observation_labels()
    
    # Store sensitivity matrices from each seed
    all_sensitivities = []
    
    # Define a consistent set of commands to use for all seeds
    # This ensures we're measuring variation due to seeds, not commands
    fixed_commands = [
        {"x": 1.0, "y": 0.0, "yaw": 0.0},   # Forward
        {"x": 0.0, "y": 1.0, "yaw": 0.0},   # Lateral
        {"x": 0.0, "y": 0.0, "yaw": 1.0},   # Turn
        {"x": 0.7, "y": 0.7, "yaw": 0.0},   # Diagonal
        {"x": -0.7, "y": 0.0, "yaw": 0.0},  # Backward
    ]
    
    # Run analysis for each seed
    for i, seed in enumerate(seeds):
        print(f"\n--- Running analysis with seed {seed} ({i+1}/{len(seeds)}) ---\n")
        
        # Create a directory for this seed's results
        seed_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Create analyzer for this seed
        analyzer = PolicyAnalyzer(
            policy=policy,
            obs_labels=obs_labels,
            action_labels=env.get_action_labels() if hasattr(env, "get_action_labels") else None,
            device=device,
            save_dir=seed_dir
        )
        
        # Set the random seed before collecting observations
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Collect observations with fixed sequence of commands
        print(f"Collecting {num_samples} observations with seed {seed}...")
        
        # Reset environment to ensure consistency
        env.reset()
        obs, extras = env.get_observations()
        
        observations = []
        for j in range(num_samples):
            # Use the same command sequence for each seed
            cmd = fixed_commands[j % len(fixed_commands)]
            
            # Apply command to observation
            obs_copy = obs.clone()
            obs_copy[:, 6] = cmd["x"]
            obs_copy[:, 7] = cmd["y"]
            obs_copy[:, 8] = cmd["yaw"]
            
            # Get policy action
            with torch.no_grad():
                action = policy(obs_copy)
            
            # Step environment
            result = env.step(action)
            
            # Handle different return formats
            if len(result) == 4:
                obs, _, _, _ = result
            elif len(result) == 5:
                obs, _, _, _, _ = result
            else:
                obs = result[0]
            
            # Store observation
            observations.append(obs_copy[0].clone())
            
            # Progress
            if (j+1) % (num_samples // 10) == 0:
                print(f"Collected {j+1}/{num_samples} samples")
        
        # Stack observations
        observations = torch.stack(observations).to(device)
        
        # Run the analysis for this seed
        analyzer.run_full_analysis(observations, seed=seed)
        
        # Extract sensitivity matrix and store it
        if analyzer.normalized_sensitivity_matrix is not None:
            sensitivity = torch.norm(analyzer.normalized_sensitivity_matrix, dim=1).cpu().numpy()
            all_sensitivities.append(sensitivity)
    
    # If we have results from multiple seeds, calculate statistics
    if len(all_sensitivities) > 0:
        # Stack all sensitivities to create a 2D array (seed × feature)
        sensitivities_array = np.stack(all_sensitivities)
        
        # Calculate statistics across seeds
        mean_importance = np.mean(sensitivities_array, axis=0)
        std_importance = np.std(sensitivities_array, axis=0)
        min_importance = np.min(sensitivities_array, axis=0)
        max_importance = np.max(sensitivities_array, axis=0)
        
        # Create aggregated analyzer for combined results
        aggregate_analyzer = PolicyAnalyzer(
            policy=policy,
            obs_labels=obs_labels,
            action_labels=env.get_action_labels() if hasattr(env, "get_action_labels") else None,
            device=device,
            save_dir=save_dir
        )
        
        # Create feature importance plot with variance bars
        plot_feature_importance_with_variance(
            obs_labels,
            mean_importance, 
            std_importance,
            min_importance,
            max_importance,
            save_dir,
            seeds=seeds
        )
        
        return aggregate_analyzer
    
    return None

def detect_gait_phases_from_contacts(self, observations, foot_contacts, num_phases=8):
    """
    More accurate gait phase detection using foot contact information.
    
    Args:
        observations: Tensor of observations [num_samples, obs_dim]
        foot_contacts: Tensor of foot contacts [num_samples, num_feet]
        num_phases: Number of phases to identify
        
    Returns:
        phase_indices: Dictionary mapping phase number to observation indices
    """
    print(f"Detecting gait phases from foot contacts with shape {foot_contacts.shape}")
    
    # Convert to numpy for easier processing
    contacts_np = foot_contacts.cpu().numpy()
    num_feet = contacts_np.shape[1]
    
    # Check if we have enough data
    if len(observations) < 10:
        print("Warning: Not enough observations to detect gait phases.")
        return {0: np.arange(len(observations))}
    
    # Calculate which foot is the reference limb (e.g., front right)
    ref_limb = 0  # Front right foot by convention
    
    # Calculate contact transitions for reference foot
    ref_contacts = contacts_np[:, ref_limb]
    contact_changes = np.diff(ref_contacts)
    
    # Detect touchdown events (contact transitions from 0 to 1)
    # and liftoff events (contact transitions from 1 to 0)
    touchdown_indices = np.where(contact_changes > 0.5)[0] + 1
    liftoff_indices = np.where(contact_changes < -0.5)[0] + 1
    
    print(f"Found {len(touchdown_indices)} touchdowns and {len(liftoff_indices)} liftoffs")
    
    # If not enough gait cycles detected, fall back to time-based
    if len(touchdown_indices) < 2:
        print("Warning: Not enough gait cycles detected. Falling back to time-based phases.")
        try:
            # Call detect_gait_phases as an instance method (with self)
            return self.detect_gait_phases(observations, None, num_phases)
        except Exception as e:
            print(f"Warning: Error in detect_gait_phases: {e}")
            # Simple time-based phase detection
            phases = np.linspace(0, num_phases, len(observations), endpoint=False).astype(int)
            phases = phases % num_phases
            
            # Group by phase
            phase_indices = {}
            for phase in range(num_phases):
                phase_indices[phase] = np.where(phases == phase)[0]
            return phase_indices
    
    # Identify phases within each stride
    phases = np.zeros(len(observations), dtype=int)
    
    for i in range(len(observations)):
        # Find which stride this observation belongs to
        stride_idx = np.searchsorted(touchdown_indices, i) - 1
        if stride_idx >= 0 and stride_idx < len(touchdown_indices) - 1:
            stride_start = touchdown_indices[stride_idx]
            stride_end = touchdown_indices[stride_idx + 1]
            
            # Calculate phase within this stride (0 to num_phases-1)
            stride_progress = (i - stride_start) / (stride_end - stride_start)
            phases[i] = int(stride_progress * num_phases) % num_phases
        else:
            # For observations before first touchdown or after last one
            # Try to estimate phase from nearest stride
            if stride_idx < 0 and len(touchdown_indices) > 0:
                # Before first touchdown
                phases[i] = (num_phases - int((touchdown_indices[0] - i) / 
                                           (touchdown_indices[0] / num_phases))) % num_phases
            elif stride_idx >= len(touchdown_indices) - 1 and len(touchdown_indices) > 1:
                # After last touchdown
                stride_len = touchdown_indices[-1] - touchdown_indices[-2]
                phases[i] = int((i - touchdown_indices[-1]) / (stride_len / num_phases)) % num_phases
            else:
                # Default to phase 0
                phases[i] = 0
    
    # Group observation indices by phase
    phase_indices = {}
    for phase in range(num_phases):
        phase_indices[phase] = np.where(phases == phase)[0]
        print(f"Phase {phase}: {len(phase_indices[phase])} samples")
    
    return phase_indices

def plot_feature_importance_with_variance(obs_labels, mean_importance, std_importance, 
                                         min_importance, max_importance, save_dir, seeds=None):
    """
    Create a feature importance plot with variance bars.
    
    Args:
        obs_labels: Labels for the observation features
        mean_importance: Mean importance values
        std_importance: Standard deviation of importance
        min_importance: Minimum importance values
        max_importance: Maximum importance values
        save_dir: Directory to save the plot
        seeds: List of seeds used for analysis
    """
    
    # Sort features by mean importance (highest to lowest)
    sorted_indices = np.argsort(-mean_importance)
    
    # Limit to top features for better visualization
    top_k = min(40, len(sorted_indices))
    top_indices = sorted_indices[:top_k]
    
    # Extract values for top features
    top_labels = [f"{obs_labels[i]} (#{i})" for i in top_indices]
    top_means = mean_importance[top_indices]
    top_stds = std_importance[top_indices]
    top_mins = min_importance[top_indices]
    top_maxes = max_importance[top_indices]
    
    # Create figure
    plt.figure(figsize=(14, max(10, top_k * 0.3)))
    
    # Create horizontal bar chart
    y_pos = np.arange(top_k)
    
    # Plot the bars with mean values
    bars = plt.barh(y_pos, top_means, align='center', alpha=0.7, 
                   color=plt.cm.viridis(np.linspace(0.1, 0.9, top_k)))
    
    # Add min-max range as BLACK lines
    for i in range(top_k):
        # Horizontal line showing min-max range
        plt.plot([top_mins[i], top_maxes[i]], [y_pos[i], y_pos[i]], 'k-', linewidth=2.5, alpha=0.8)
        # Vertical caps at min and max
        plt.plot([top_mins[i]], [y_pos[i]], 'k|', markersize=10)
        plt.plot([top_maxes[i]], [y_pos[i]], 'k|', markersize=10)
    
    # Add standard deviation indicators in RED
    plt.errorbar(top_means, y_pos, xerr=top_stds, fmt='none', ecolor='red', 
                capsize=5, alpha=0.8, label='±1 Std Dev')
    
    # Add value labels with standard deviation
    for i in range(top_k):
        plt.text(top_means[i] + 0.01, y_pos[i], 
                f'{top_means[i]:.3f} ±{top_stds[i]:.3f}', 
                va='center', fontsize=9)
    
    # Improve aesthetics
    plt.yticks(y_pos, top_labels)
    plt.xlabel('Feature Importance (Mean Across Seeds)', fontweight='bold')
    plt.title('Feature Importance Ranking with Variance Across Seeds', 
             fontsize=14, fontweight='bold')
    
    # Add legend for clarity
    plt.plot([], [], 'k-', label='Min-Max Range')
    plt.legend(loc='lower right')
    
    # Add seeds information as text
    if seeds:
        plt.figtext(0.02, 0.01, f"Based on {len(seeds)} seeds: {', '.join(map(str, seeds))}", 
                   fontsize=8)
    
    # Add grid for easier reading
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure with clear naming
    plt.savefig(f"{save_dir}/feature_importance_with_variance.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/feature_importance_with_variance.pdf", bbox_inches='tight')
    
    # Print confirmation
    print(f"Saved feature importance plot with variance bars to:")
    print(f"  - {save_dir}/feature_importance_with_variance.png")
    print(f"  - {save_dir}/feature_importance_with_variance.pdf")
    
    # Also save the data in CSV format for further analysis
    with open(f"{save_dir}/feature_importance_with_variance.csv", 'w') as f:
        f.write("feature,feature_id,mean,std,min,max\n")
        for idx in sorted_indices:
            name = obs_labels[idx] if idx < len(obs_labels) else f"Feature_{idx}"
            f.write(f"{name},{idx},{mean_importance[idx]:.6f},{std_importance[idx]:.6f},"
                   f"{min_importance[idx]:.6f},{max_importance[idx]:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate feature importance across runs')
    parser.add_argument('--log_folder', type=str, required=True,
                        help='Path to the folder containing training runs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save aggregated results')
    
    args = argparse.ArgumentParser()
    
    # Run the aggregation
    PolicyAnalyzer.aggregate_feature_importance(args.log_folder, args.output_dir)