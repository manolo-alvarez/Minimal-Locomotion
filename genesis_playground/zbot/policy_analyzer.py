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

    def phase_aware_sensitivity_analysis(self, observations, env, num_phases=8, perturbation_scale=0.01, phase_indices=None):
        """
        Analyze feature sensitivity across different phases of the gait cycle.
        
        Args:
            observations: Collected observations tensor
            env: Environment instance for gait phase detection
            num_phases: Number of phases to divide the gait cycle into (if detecting phases)
            perturbation_scale: Scale of perturbations for sensitivity analysis
            phase_indices: Pre-computed phase indices (dictionary mapping phase number to observation indices)
        """
        # First collect overall statistics
        self.collect_statistics(observations)
        
        # Detect gait phases if not provided
        if phase_indices is None:
            print(f"Detecting {num_phases} gait phases...")
            phase_indices = detect_gait_phases(observations, env, num_phases)
        else:
            # Use provided phases, but update num_phases if needed
            num_phases = max(phase_indices.keys()) + 1
            print(f"Using pre-computed gait phases ({num_phases} phases)")
        
        # Initialize storage for phase-specific sensitivities
        phase_sensitivities = {}
        
        # Create a subplot grid for phase-specific importance plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True) if num_phases == 8 else plt.subplots(
            int(np.ceil(num_phases/4)), 4, figsize=(20, 5*np.ceil(num_phases/4)), sharey=True)
        axes = axes.flatten()
        
        # Analyze each phase separately
        for phase in range(num_phases):
            idx = phase_indices[phase]
            if len(idx) < 5:  # Skip phases with too few samples
                print(f"Warning: Phase {phase} has only {len(idx)} samples. Skipping.")
                continue
                
            print(f"Analyzing phase {phase}/{num_phases} with {len(idx)} samples")
            
            # Get representative observation for this phase
            phase_obs = observations[idx]
            base_obs = phase_obs.mean(dim=0, keepdim=True)
            
            # Get action dimension by doing a forward pass
            with torch.no_grad():
                test_action = self.policy(base_obs)
            action_dim = test_action.shape[1]
            
            # Initialize sensitivity matrices for this phase
            obs_dim = observations.shape[1]
            sensitivity = torch.zeros((obs_dim, action_dim), device=self.device)
            
            # Compute sensitivity using central difference method
            for i in range(obs_dim):
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
                sensitivity[i, :] = feature_sensitivity
            
            # Store this phase's sensitivity
            phase_sensitivities[phase] = sensitivity
            
            # Plot feature importance for this phase
            feature_importance = torch.norm(sensitivity, dim=1).cpu().numpy()
            sorted_indices = np.argsort(-feature_importance)[:20]  # Top 20 features
            
            # Plot in the appropriate subplot
            ax = axes[phase]
            labels = [self.obs_labels[i] for i in sorted_indices]
            values = feature_importance[sorted_indices]
            
            ax.barh(range(len(labels)), values, align='center', color='teal')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"Phase {phase}", fontweight='bold')
            if phase % 4 == 0:
                ax.set_ylabel("Features")
            if phase >= num_phases-4:
                ax.set_xlabel("Importance")
        
        # Hide any unused subplots
        for i in range(num_phases, len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/phase_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap showing feature importance across phases
        self._plot_phase_importance_heatmap(phase_sensitivities, num_phases)
        
        return phase_sensitivities

    def _plot_phase_importance_heatmap(self, phase_sensitivities, num_phases):
        """Plot heatmap of feature importance across different gait phases."""
        if not phase_sensitivities:
            return
            
        # Get feature dimension from first phase
        first_phase = list(phase_sensitivities.keys())[0]
        obs_dim = phase_sensitivities[first_phase].shape[0]
        
        # Prepare data for heatmap (features x phases)
        heatmap_data = np.zeros((obs_dim, num_phases))
        
        # Calculate importance for each feature in each phase
        for phase in range(num_phases):
            if phase in phase_sensitivities:
                # Calculate importance as L2 norm across actions
                importance = torch.norm(phase_sensitivities[phase], dim=1).cpu().numpy()
                heatmap_data[:, phase] = importance
        
        # Sort features by overall importance
        overall_importance = heatmap_data.sum(axis=1)
        sorted_indices = np.argsort(-overall_importance)
        
        # Select top 25 features for better visualization
        top_indices = sorted_indices[:25]
        top_data = heatmap_data[top_indices, :]
        top_labels = [self.obs_labels[i] for i in top_indices]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        if HAS_SEABORN:
            sns.heatmap(top_data, cmap='viridis', 
                       xticklabels=range(num_phases),
                       yticklabels=top_labels,
                       cbar_kws={'label': 'Feature Importance'})
        else:
            plt.imshow(top_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Feature Importance')
            plt.yticks(np.arange(len(top_labels)), top_labels)
            plt.xticks(np.arange(num_phases), range(num_phases))
        
        plt.title("Feature Importance Across Gait Phases", fontsize=14, fontweight='bold')
        plt.xlabel("Gait Phase", fontweight='bold')
        plt.ylabel("Feature", fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f"{self.save_dir}/phase_importance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def detect_gait_phases_from_contacts(self, observations, foot_contacts, num_phases=8):
        """More accurate gait phase detection using foot contact information."""
        # Assuming foot_contacts is a tensor [num_samples, num_feet]
        num_feet = foot_contacts.shape[1]
        
        # Detect gait cycles using foot contact patterns
        # For a quadruped, typically one full gait cycle happens when all feet complete 
        # their stance-swing-stance pattern
        
        # Example for typical quadruped gaits:
        # 1. Calculate which foot is the reference limb (e.g., front right)
        ref_limb = 0  # Front right foot
        
        # 2. Detect touchdown events (contact transitions from 0 to 1)
        contact_changes = foot_contacts[1:, ref_limb] - foot_contacts[:-1, ref_limb]
        touchdown_indices = torch.where(contact_changes == 1)[0] + 1
        
        # 3. Each pair of consecutive touchdowns defines one stride
        if len(touchdown_indices) < 2:
            print("Warning: Not enough gait cycles detected. Falling back to time-based phases.")
            # Fall back to time-based phases
            return detect_gait_phases(observations, None, num_phases)
        
        # 4. Identify phases within each stride
        phases = []
        for i in range(len(observations)):
            # Find which stride this observation belongs to
            stride_idx = np.searchsorted(touchdown_indices, i) - 1
            if stride_idx >= 0 and stride_idx < len(touchdown_indices) - 1:
                stride_start = touchdown_indices[stride_idx]
                stride_end = touchdown_indices[stride_idx + 1]
                # Calculate phase within this stride (0 to num_phases-1)
                stride_progress = (i - stride_start) / (stride_end - stride_start)
                phase = int(stride_progress * num_phases) % num_phases
            else:
                # For observations before first touchdown or after last one
                phase = 0
            phases.append(phase)
        
        # Group observation indices by phase
        phase_indices = {}
        for phase in range(num_phases):
            phase_indices[phase] = np.where(np.array(phases) == phase)[0]
        
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
    
    # Plot aggregated results
    # ...

def detect_gait_phases_from_contacts(observations, foot_contacts, num_phases=8):
    """More accurate gait phase detection using foot contact information."""
    # Assuming foot_contacts is a tensor [num_samples, num_feet]
    num_feet = foot_contacts.shape[1]
    
    # Detect gait cycles using foot contact patterns
    # For a quadruped, typically one full gait cycle happens when all feet complete 
    # their stance-swing-stance pattern
    
    # Example for typical quadruped gaits:
    # 1. Calculate which foot is the reference limb (e.g., front right)
    ref_limb = 0  # Front right foot
    
    # 2. Detect touchdown events (contact transitions from 0 to 1)
    contact_changes = foot_contacts[1:, ref_limb] - foot_contacts[:-1, ref_limb]
    touchdown_indices = torch.where(contact_changes == 1)[0] + 1
    
    # 3. Each pair of consecutive touchdowns defines one stride
    if len(touchdown_indices) < 2:
        print("Warning: Not enough gait cycles detected. Falling back to time-based phases.")
        # Fall back to time-based phases
        return detect_gait_phases(observations, None, num_phases)
    
    # 4. Identify phases within each stride
    phases = []
    for i in range(len(observations)):
        # Find which stride this observation belongs to
        stride_idx = np.searchsorted(touchdown_indices, i) - 1
        if stride_idx >= 0 and stride_idx < len(touchdown_indices) - 1:
            stride_start = touchdown_indices[stride_idx]
            stride_end = touchdown_indices[stride_idx + 1]
            # Calculate phase within this stride (0 to num_phases-1)
            stride_progress = (i - stride_start) / (stride_end - stride_start)
            phase = int(stride_progress * num_phases) % num_phases
        else:
            # For observations before first touchdown or after last one
            phase = 0
        phases.append(phase)
    
    # Group observation indices by phase
    phase_indices = {}
    for phase in range(num_phases):
        phase_indices[phase] = np.where(np.array(phases) == phase)[0]
    
    return phase_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate feature importance across runs')
    parser.add_argument('--log_folder', type=str, required=True,
                        help='Path to the folder containing training runs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save aggregated results')
    
    args = parser.parse_args()
    
    # Run the aggregation
    PolicyAnalyzer.aggregate_feature_importance(args.log_folder, args.output_dir)