""" Feature importance analysis for ZBot policy. """
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
import os

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
        
    def collect_statistics(self, observations: torch.Tensor, actions: torch.Tensor):
        """
        Collect statistics about observations and actions.
        
        Args:
            observations: Tensor of observations [num_samples, obs_dim]
            actions: Tensor of actions [num_samples, action_dim]
        """
        # Compute statistics for observations
        self.obs_stats['mean'] = observations.mean(dim=0)
        self.obs_stats['std'] = observations.std(dim=0)
        self.obs_stats['min'] = observations.min(dim=0)[0]
        self.obs_stats['max'] = observations.max(dim=0)[0]
        
        # Compute statistics for actions
        self.action_stats['mean'] = actions.mean(dim=0)
        self.action_stats['std'] = actions.std(dim=0)
        self.action_stats['min'] = actions.min(dim=0)[0]
        self.action_stats['max'] = actions.max(dim=0)[0]
        
    def compute_sensitivity(
        self, 
        base_observation: torch.Tensor,
        perturbation_scale: float = 0.01,
        num_samples: int = 10
    ):
        """
        Compute sensitivity of policy outputs to input perturbations.
        
        Args:
            base_observation: Base observation to perturb [obs_dim]
            perturbation_scale: Scale of perturbations relative to feature std
            num_samples: Number of samples for each feature perturbation
        """
        base_observation = base_observation.to(self.device)
        obs_dim = base_observation.shape[0]
        action_dim = len(self.action_labels)
        
        # If we don't have statistics yet, use simple perturbation
        if self.obs_stats['std'] is None:
            perturbation_sizes = torch.ones(obs_dim, device=self.device) * perturbation_scale
        else:
            # Scale perturbation by feature std to account for different scales
            perturbation_sizes = self.obs_stats['std'] * perturbation_scale
        
        # Get baseline action
        with torch.no_grad():
            base_action = self.policy(base_observation.unsqueeze(0)).squeeze(0)
        
        # Initialize sensitivity matrices
        self.sensitivity_matrix = torch.zeros((obs_dim, action_dim), device=self.device)
        self.normalized_sensitivity_matrix = torch.zeros((obs_dim, action_dim), device=self.device)
        
        # Compute sensitivity for each feature
        for i in range(obs_dim):
            perturbations = torch.zeros(num_samples, obs_dim, device=self.device)
            perturbations[:, i] = torch.linspace(-perturbation_sizes[i], 
                                                perturbation_sizes[i], 
                                                num_samples)
            
            # Create perturbed observations
            perturbed_obs = base_observation.unsqueeze(0) + perturbations
            
            # Get actions for perturbed observations
            with torch.no_grad():
                perturbed_actions = self.policy(perturbed_obs)
            
            # Compute sensitivity as average gradient of action w.r.t. observation
            # Using central difference approximation
            feature_sensitivity = (perturbed_actions[-1] - perturbed_actions[0]) / (2 * perturbation_sizes[i])
            self.sensitivity_matrix[i] = feature_sensitivity
            
            # Normalize sensitivity by typical range of action values
            if self.action_stats['std'] is not None:
                action_ranges = self.action_stats['max'] - self.action_stats['min']
                # Avoid division by zero
                action_ranges = torch.where(action_ranges > 1e-6, action_ranges, torch.ones_like(action_ranges))
                
                # Scale sensitivity by obs_std / action_range
                self.normalized_sensitivity_matrix[i] = (
                    self.sensitivity_matrix[i] * self.obs_stats['std'][i] / action_ranges
                )
            else:
                self.normalized_sensitivity_matrix = self.sensitivity_matrix.clone()
                
    def plot_feature_statistics(self):
        """Plot statistics for each feature."""
        if self.obs_stats['mean'] is None:
            print("No statistics collected. Run collect_statistics first.")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Extract data
        means = self.obs_stats['mean'].cpu().numpy()
        stds = self.obs_stats['std'].cpu().numpy()
        mins = self.obs_stats['min'].cpu().numpy()
        maxs = self.obs_stats['max'].cpu().numpy()
        
        # Sort by std
        indices = np.argsort(stds)[::-1]  # Sort by standard deviation (descending)
        sorted_labels = [self.obs_labels[i] for i in indices]
        
        x = np.arange(len(sorted_labels))
        plt.errorbar(x, means[indices], yerr=stds[indices], fmt='o', label='Mean ± Std')
        plt.scatter(x, mins[indices], marker='_', color='red', label='Min')
        plt.scatter(x, maxs[indices], marker='_', color='green', label='Max')
        
        plt.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.title('Feature Statistics (Sorted by Standard Deviation)')
        plt.xticks(x, sorted_labels, rotation=90)
        plt.legend()
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
        
        # Choose sensitivity matrix
        sensitivity = self.normalized_sensitivity_matrix if normalized else self.sensitivity_matrix
        
        # Calculate overall feature importance as L2 norm across actions
        feature_importance = torch.norm(sensitivity, dim=1).cpu().numpy()
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[::-1]
        top_k = min(top_k, len(sorted_indices))  # Ensure we don't exceed number of features
        top_indices = sorted_indices[:top_k]
        
        plt.figure(figsize=(10, 8))
        
        # Get feature labels and importance values
        labels = [self.obs_labels[i] for i in top_indices]
        values = feature_importance[top_indices]
        
        # Create horizontal bar chart
        plt.barh(np.arange(len(labels)), values, align='center')
        plt.yticks(np.arange(len(labels)), labels)
        plt.xlabel('Sensitivity (L2 norm across actions)')
        plt.title(f'Top {top_k} Features by {"Normalized " if normalized else ""}Sensitivity')
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
        top_k_features = min(top_k_features, len(feature_importance))  # Ensure we don't exceed number of features
        feature_indices = np.argsort(feature_importance)[::-1][:top_k_features]
        
        # Calculate overall action sensitivity
        action_sensitivity = np.linalg.norm(abs_sensitivity, axis=0)
        if top_k_actions is not None:
            top_k_actions = min(top_k_actions, len(action_sensitivity))  # Ensure we don't exceed number of actions
            action_indices = np.argsort(action_sensitivity)[::-1][:top_k_actions]
        else:
            action_indices = np.arange(len(self.action_labels))
            
        # Extract subset of sensitivity matrix
        subset_sensitivity = abs_sensitivity[np.ix_(feature_indices, action_indices)]
        
        plt.figure(figsize=(12, 10))
        
        # Use either seaborn or matplotlib for the heatmap
        if HAS_SEABORN:
            ax = sns.heatmap(
                subset_sensitivity,
                cmap='viridis',
                xticklabels=[self.action_labels[i] for i in action_indices],
                yticklabels=[self.obs_labels[i] for i in feature_indices],
                cbar_kws={'label': 'Absolute Normalized Sensitivity'}
            )
        else:
            # Matplotlib alternative to seaborn heatmap
            plt.imshow(subset_sensitivity, cmap='viridis', aspect='auto')
            plt.colorbar(label='Absolute Normalized Sensitivity')
            
            # Set x and y ticks
            plt.yticks(np.arange(len(feature_indices)), 
                      [self.obs_labels[i] for i in feature_indices])
            plt.xticks(np.arange(len(action_indices)), 
                      [self.action_labels[i] for i in action_indices], 
                      rotation=45, ha='right')
        
        plt.title('Feature-Action Sensitivity Heatmap')
        plt.tight_layout()
        
        plt.savefig(f"{self.save_dir}/feature_action_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_analysis(
        self, 
        observations: torch.Tensor,
        base_observation: Optional[torch.Tensor] = None,
        perturbation_scale: float = 0.01,
        top_k: int = 20
    ):
        """
        Run full feature importance analysis.
        
        Args:
            observations: Tensor of observations [num_samples, obs_dim]
            base_observation: Observation to use for sensitivity analysis (uses mean if None)
            perturbation_scale: Scale of perturbations
            top_k: Number of top features to show
        """
        print("Running policy feature analysis...")
        
        # Collect action statistics
        with torch.no_grad():
            actions = self.policy(observations)
        
        # Collect statistics
        print("Collecting statistics...")
        self.collect_statistics(observations, actions)
        
        # Use mean observation if base_observation not provided
        if base_observation is None:
            base_observation = self.obs_stats['mean']
            
        # Compute sensitivity
        print("Computing sensitivity...")
        self.compute_sensitivity(base_observation, perturbation_scale)
        
        # Generate plots
        print("Generating plots...")
        self.plot_feature_statistics()
        self.plot_top_features_by_sensitivity(top_k=top_k, normalized=True)
        self.plot_top_features_by_sensitivity(top_k=top_k, normalized=False)
        self.plot_feature_action_heatmap(top_k_features=top_k)
        
        print(f"Analysis complete. Results saved to {self.save_dir}/")
        
        # Return top feature indices for potential use elsewhere
        sensitivity = torch.norm(self.normalized_sensitivity_matrix, dim=1).cpu().numpy()
        return np.argsort(sensitivity)[::-1]


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
    
    # Create meaningful observation labels
    # These are examples based on your keyboard control function
    # Adjust these based on your actual observation space
    base_obs_labels = [
        "root_h",          # Root height
        "root_roll",       # Root roll orientation
        "root_pitch",      # Root pitch orientation
        "root_yaw",        # Root yaw orientation
        "root_vx",         # Root x velocity
        "root_vy",         # Root y velocity
        "command_x",       # Forward command
        "command_y",       # Lateral command 
        "command_yaw"      # Turn command
    ]
    
    # Fill remaining observation dimensions with generic labels
    obs_labels = base_obs_labels.copy()
    if obs_dim > len(obs_labels):
        obs_labels.extend([f"Feature_{i+len(base_obs_labels)}" for i in range(obs_dim - len(base_obs_labels))])
    
    # Create action labels
    with torch.no_grad():
        test_action = policy(obs)
    action_dim = test_action.shape[1]
    action_labels = [f"Joint_{i}" for i in range(action_dim)]
    
    print(f"Observation space: {obs_dim} dimensions")
    print(f"Action space: {action_dim} dimensions")
    
    # Create analyzer
    analyzer = PolicyAnalyzer(
        policy=policy,
        obs_labels=obs_labels,
        action_labels=action_labels,
        device=device,
        save_dir=save_dir
    )
    
    # Collect samples
    print(f"Collecting {num_samples} samples for analysis...")
    observations = []
    
    # Reset environment
    obs, _ = env.get_observations()
    
    # Use fixed commands for consistent evaluation
    fixed_cmd = {"x": 1.0, "y": 0.0, "yaw": 0.0}
    
    for i in range(num_samples):
        # Apply fixed command to observation
        obs_copy = obs.clone()
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