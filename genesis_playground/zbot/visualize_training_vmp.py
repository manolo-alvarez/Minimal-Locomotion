""" Train ZBot with VMP conditioning """
# Add these imports at the VERY TOP of zbot_train_vmp.py
import sys
import os

# Add PyTorch optimization flags
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from custom_policies import VAEConditionedActorCritic
import os
import argparse
import pickle
import shutil
import wandb
import torch
import numpy as np
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from zbot_env_vmp import ZbotEnvVMP
import genesis as gs
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.storage import RolloutStorage
import matplotlib.pyplot as plt
from glob import glob

# Make VAEConditionedActorCritic available globally
globals()['VAEConditionedActorCritic'] = VAEConditionedActorCritic

def get_train_cfg(exp_name, max_iterations, latent_dim):
    """ Unified training config including original and VMP parameters """
    return {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.02,
            "entropy_coef": 0.02,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "class_name": "PPO",
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "VAEConditionedActorCritic",
            "conditional_dim": latent_dim + 20  # VMP addition
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 48,
            "policy_class_name": "VAEConditionedActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "OnPolicyRunner",
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
        "num_steps_per_env": 48,
        "save_interval": 100,
        "empirical_normalization": True,
        "vmp_params": {  # New VMP-specific parameters
            "window_size": 15,
            "latent_dim": 64,
            "motion_dim": 20
        }
    }

def get_cfgs(vae_model_path=None):
    """ Unified configuration combining original and VMP parameters """
    # Original parameters
    env_cfg = {
        "num_actions": 10,
        "default_joint_angles": {
            "R_Hip_Pitch": 0.0,
            "L_Hip_Pitch": 0.0,
            "R_Hip_Yaw": 0.0,
            "L_Hip_Yaw": 0.0,
            "R_Hip_Roll": 0.0,
            "L_Hip_Roll": 0.0,
            "R_Knee_Pitch": 0.0,
            "L_Knee_Pitch": 0.0,
            "R_Ankle_Pitch": 0.0,
            "L_Ankle_Pitch": 0.0,
        },
        "dof_names": [
            "R_Hip_Pitch", "L_Hip_Pitch", "R_Hip_Yaw", "L_Hip_Yaw",
            "R_Hip_Roll", "L_Hip_Roll", "R_Knee_Pitch", "L_Knee_Pitch",
            "R_Ankle_Pitch", "L_Ankle_Pitch",
        ],
        "env_friction_range": {"start": [0.9, 1.1], "end": [0.9, 1.1]},
        "link_mass_multipliers": {"start": [1.0, 1.0], "end": [1.0, 1.0]},
        "rfi_scale": 0.1,
        "kp": 20.0,
        "kd": 0.5,
        "kp_multipliers": [0.75, 1.25],
        "kd_multipliers": [0.75, 1.25],
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.41],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "max_torque": 10.0,
        # VMP additions
        "selected_joints": [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee',
            'right_ankle', 'left_ankle'
        ],
        "window_size": 15
    }

    obs_cfg = {
        "num_obs": 39 + 64 + 20,  # Original + latent + motion
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "motion": 1.0,  # VMP addition
            "latent": 1.0   # VMP addition
        },
        "obs_exclusions": []
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "base_height": -50.0,
            "action_rate": -0.01,
            "similar_to_default": -0.2,
            "feet_air_time": 10.0,
            "motion_tracking": 2.0
        },
        # VMP additions
        "pos_weight": 2.0,
        "vel_weight": 0.2,
        "alive_bonus": 0.5
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_y_range": [0.0, 0.0],
        "lin_vel_x_range": [-0.2, 0.4],
        "ang_vel_range": [-0.4, 0.4]
    }

    # Add VAE path if provided
    if vae_model_path:
        env_cfg["vae_model_path"] = vae_model_path

    return env_cfg, obs_cfg, reward_cfg, command_cfg

class Transition:
    def __init__(self, num_envs, num_obs, num_actions, num_critic_obs=None, device="cpu"):
        self.observations = torch.zeros(num_envs, num_obs, device=device)
        self.privileged_observations = torch.zeros(num_envs, num_critic_obs, device=device) if num_critic_obs else None
        self.actions = torch.zeros(num_envs, num_actions, device=device)
        self.rewards = torch.zeros(num_envs, device=device)
        self.dones = torch.zeros(num_envs, device=device)
        self.values = torch.zeros(num_envs, device=device)
        self.actions_log_prob = torch.zeros(num_envs, device=device)
        self.rnd_state = torch.zeros(num_envs, device=device)  # Changed to match storage expectations
        self.action_mean = torch.zeros(num_envs, num_actions, device=device)
        self.action_sigma = torch.zeros(num_envs, num_actions, device=device)
        self.hidden_states = None  # Added for RNN support, None since we're not using RNNs
        
    def clear(self):
        """Reset all tensors to zero."""
        self.observations.zero_()
        if self.privileged_observations is not None:
            self.privileged_observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()
        self.actions_log_prob.zero_()
        self.rnd_state.zero_()
        self.action_mean.zero_()
        self.action_sigma.zero_()
        self.hidden_states = None

class CustomOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Store initial parameters
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"].copy()  # Make a copy to avoid modifying original
        self.policy_cfg = train_cfg["policy"].copy()  # Make a copy to avoid modifying original
        self.device = device
        self.env = env
        self.log_dir = log_dir
        self.git_status_repos = []  # Initialize empty list for git repos
        
        # Initialize logging-related attributes that parent would set
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.writer = None
        
        # Initialize training log file
        if self.log_dir:
            self.train_log_file = open(os.path.join(self.log_dir, "training.txt"), "w")
        else:
            self.train_log_file = None

        # Get observation dimensions
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        # Initialize normalization-related attributes
        self.empirical_normalization = self.cfg.get("empirical_normalization", False)
        if self.empirical_normalization:
            from rsl_rl.modules import EmpiricalNormalization
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)

        # Extract policy parameters
        actor_hidden_dims = self.policy_cfg.get('actor_hidden_dims', [512, 256, 128])
        critic_hidden_dims = self.policy_cfg.get('critic_hidden_dims', [512, 256, 128])
        activation = self.policy_cfg.get('activation', 'elu')
        init_noise_std = self.policy_cfg.get('init_noise_std', 1.0)
        conditional_dim = self.policy_cfg.get('conditional_dim', None)

        # Create policy directly
        actor_critic = VAEConditionedActorCritic(
            num_obs=num_obs,
            num_privileged_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            conditional_dim=conditional_dim
        ).to(self.device)

        # Set other parameters that parent would set
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Initialize storage with proper shapes
        if hasattr(self.env, 'num_privileged_obs') and self.env.num_privileged_obs is not None:
            privileged_obs_shape = (self.env.num_privileged_obs,)
        else:
            privileged_obs_shape = None

        self.storage = RolloutStorage(
            num_transitions_per_env=self.num_steps_per_env,
            num_envs=self.env.num_envs,
            obs_shape=(self.env.num_obs,),
            privileged_obs_shape=privileged_obs_shape,
            actions_shape=(self.env.num_actions,),
            rnd_state_shape=(),  # Empty tuple for non-recurrent networks
            device=self.device
        )

        # Initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        # Initialize storage in the algorithm
        self.alg.storage = self.storage
        # Initialize transition object
        self.alg.transition = Transition(
            num_envs=self.env.num_envs,
            num_obs=self.env.num_obs,
            num_actions=self.env.num_actions,
            num_critic_obs=num_critic_obs,
            device=self.device
        )

    def log(self, info):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        super().log(info)
        
        # Write to training log file
        if self.train_log_file:
            log_str = f"Iteration: {self.current_learning_iteration}\n"
            
            # Log all metrics from info dict
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    # Clean up the key name for better readability
                    clean_key = key.replace('train/', '')  # Remove 'train/' prefix
                    log_str += f"{clean_key}: {value}\n"
            
            # Add reward metrics explicitly
            reward_metrics = {
                'total_reward': info.get('train/total_reward', 0),
                'episode_length': info.get('train/episode_length', 0),
                'tracking_lin_vel': info.get('train/tracking_lin_vel', 0),
                'tracking_ang_vel': info.get('train/tracking_ang_vel', 0),
                'lin_vel_z': info.get('train/lin_vel_z', 0),
                'base_height': info.get('train/base_height', 0),
                'action_rate': info.get('train/action_rate', 0),
                'similar_to_default': info.get('train/similar_to_default', 0),
                'feet_air_time': info.get('train/feet_air_time', 0),
                'motion_tracking': info.get('train/motion_tracking', 0),
            }
            
            # Add raw reward metrics
            raw_metrics = {
                'raw_tracking_lin_vel': info.get('train/raw_tracking_lin_vel', 0),
                'raw_tracking_ang_vel': info.get('train/raw_tracking_ang_vel', 0),
                'raw_lin_vel_z': info.get('train/raw_lin_vel_z', 0),
                'raw_base_height': info.get('train/raw_base_height', 0),
                'raw_action_rate': info.get('train/raw_action_rate', 0),
                'raw_similar_to_default': info.get('train/raw_similar_to_default', 0),
                'raw_feet_air_time': info.get('train/raw_feet_air_time', 0),
                'raw_motion_tracking': info.get('train/raw_motion_tracking', 0),
            }
            
            # Add all metrics to log
            for key, value in {**reward_metrics, **raw_metrics}.items():
                log_str += f"{key}: {value}\n"
            
            # Add training metrics
            training_metrics = {
                'value_loss': info.get('train/value_loss', 0),
                'policy_loss': info.get('train/policy_loss', 0),
                'entropy_loss': info.get('train/entropy_loss', 0),
                'learning_rate': info.get('train/learning_rate', 0),
                'noise_std': info.get('train/noise_std', 0),
                'mean_value': info.get('train/mean_value', 0),
            }
            
            for key, value in training_metrics.items():
                log_str += f"{key}: {value}\n"
            
            # Add separator
            log_str += "-" * 80 + "\n"
            
            # Write and flush
            self.train_log_file.write(log_str)
            self.train_log_file.flush()
            
        # Log to wandb if enabled
        if hasattr(self, 'wandb'):
            metrics = {
                'train/total_reward': info.get('train/total_reward', 0),
                'train/episode_length': info.get('train/episode_length', 0),
                'train/tracking_lin_vel': info.get('train/tracking_lin_vel', 0),
                'train/tracking_ang_vel': info.get('train/tracking_ang_vel', 0),
                'train/lin_vel_z': info.get('train/lin_vel_z', 0),
                'train/base_height': info.get('train/base_height', 0),
                'train/action_rate': info.get('train/action_rate', 0),
                'train/similar_to_default': info.get('train/similar_to_default', 0),
                'train/feet_air_time': info.get('train/feet_air_time', 0),
                'train/motion_tracking': info.get('train/motion_tracking', 0),
                'train/value_loss': info.get('train/value_loss', 0),
                'train/policy_loss': info.get('train/policy_loss', 0),
                'train/entropy_loss': info.get('train/entropy_loss', 0),
                'train/learning_rate': info.get('train/learning_rate', 0),
                'train/noise_std': info.get('train/noise_std', 0),
                'train/mean_value': info.get('train/mean_value', 0),
            }
            wandb.log(metrics)

    def __del__(self):
        # Close the training log file when the runner is destroyed
        if hasattr(self, 'train_log_file') and self.train_log_file:
            self.train_log_file.close()

def parse_training_log(log_file):
    """Parse a training log file and extract metrics."""
    metrics = {}
    current_iteration = None
    
    print(f"\nParsing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Iteration:'):
                if current_iteration is not None:
                    metrics[current_iteration] = current_metrics
                current_iteration = int(line.split(':')[1])
                current_metrics = {}
            elif ':' in line and current_iteration is not None:
                key, value = line.split(':', 1)
                try:
                    # Preserve the 'train/' prefix if it exists in the original key
                    if key.strip().startswith('train/'):
                        current_metrics[key.strip()] = float(value.strip())
                    else:
                        # Add 'train/' prefix to all metrics
                        current_metrics[f"train/{key.strip()}"] = float(value.strip())
                except ValueError:
                    continue
    
    # Add the last iteration
    if current_iteration is not None:
        metrics[current_iteration] = current_metrics
    
    # Print sample values from first iteration
    if metrics:
        first_iter = min(metrics.keys())
        print(f"\nSample values from iteration {first_iter}:")
        for metric in ['train/total_reward', 'train/tracking_lin_vel', 'train/tracking_ang_vel', 
                      'train/raw_tracking_lin_vel', 'train/raw_tracking_ang_vel']:
            if metric in metrics[first_iter]:
                print(f"{metric}: {metrics[first_iter][metric]}")
    
    return metrics

def aggregate_metrics(log_dir):
    """Aggregate metrics from multiple training runs."""
    # Find all training.txt files in the log directory
    log_files = glob(os.path.join(log_dir, "training.txt"))
    
    if not log_files:
        raise ValueError(f"No training log files found in {log_dir}")
    
    print(f"\nFound {len(log_files)} log files in {log_dir}")
    
    # Parse metrics from each log file
    all_metrics = {}
    for log_file in log_files:
        metrics = parse_training_log(log_file)
        for iteration, iteration_metrics in metrics.items():
            if iteration not in all_metrics:
                all_metrics[iteration] = []
            all_metrics[iteration].append(iteration_metrics)
    
    # Aggregate metrics across runs
    aggregated = {}
    for iteration, iteration_metrics_list in all_metrics.items():
        aggregated[iteration] = {}
        for metric in iteration_metrics_list[0].keys():
            values = [m[metric] for m in iteration_metrics_list if metric in m]
            if values:
                aggregated[iteration][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    
    # Print sample aggregated values
    if aggregated:
        first_iter = min(aggregated.keys())
        print(f"\nAggregated values from iteration {first_iter}:")
        for metric in ['train/total_reward', 'train/tracking_lin_vel', 'train/tracking_ang_vel', 
                      'train/raw_tracking_lin_vel', 'train/raw_tracking_ang_vel']:
            if metric in aggregated[first_iter]:
                print(f"{metric}: mean={aggregated[first_iter][metric]['mean']}, std={aggregated[first_iter][metric]['std']}")
    
    return aggregated

def plot_aggregated_metrics(aggregated_metrics, output_dir=None, log_folder=None):
    """Plot aggregated metrics with variance."""
    iterations = sorted(aggregated_metrics.keys())
    
    # Print available metrics for debugging
    if iterations:
        print("\nAvailable metrics in first iteration:")
        for metric in aggregated_metrics[iterations[0]].keys():
            print(f"- {metric}")
        
        # Print sample values for key metrics
        print("\nSample values for key metrics:")
        for metric in ['train/value_loss', 'train/policy_loss', 'train/entropy_loss', 
                      'train/rew_tracking_lin_vel', 'train/rew_tracking_ang_vel']:
            if metric in aggregated_metrics[iterations[0]]:
                print(f"{metric}: {aggregated_metrics[iterations[0]][metric]['mean']}")
    
    # Set style for better aesthetics
    plt.style.use('bmh')  # Use a more modern style
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4)
    
    # Plot total reward
    ax = fig.add_subplot(gs[0, 0])
    total_reward_metrics = ['train/rew_tracking_lin_vel', 'train/rew_tracking_ang_vel']
    print("\nTotal reward metrics:")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, and green
    line_styles = ['-', '--', '-.']  # Different line styles for better distinction
    
    # Plot individual tracking rewards
    for idx, metric in enumerate(total_reward_metrics):
        means = [aggregated_metrics[i][metric]['mean'] for i in iterations if metric in aggregated_metrics[i]]
        stds = [aggregated_metrics[i][metric]['std'] for i in iterations if metric in aggregated_metrics[i]]
        if means:
            print(f"{metric}: {len(means)} points, first value: {means[0]}")
            label = metric.replace('train/rew_', '').replace('_', ' ').title()
            ax.plot(iterations[:len(means)], means, label=label, 
                   color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
            ax.fill_between(iterations[:len(means)], 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color=colors[idx], alpha=0.15)
    
    # Plot mean total reward (sum of tracking rewards)
    total_reward_means = []
    total_reward_stds = []
    for i in iterations:
        rewards = []
        stds = []
        for metric in total_reward_metrics:
            if metric in aggregated_metrics[i]:
                rewards.append(aggregated_metrics[i][metric]['mean'])
                stds.append(aggregated_metrics[i][metric]['std'])
        if rewards:
            total_reward_means.append(sum(rewards))
            total_reward_stds.append(np.sqrt(sum(s**2 for s in stds)))
    
    if total_reward_means:
        ax.plot(iterations[:len(total_reward_means)], total_reward_means, 
                label='Total Reward', color=colors[2], linewidth=3, linestyle=line_styles[2])
        ax.fill_between(iterations[:len(total_reward_means)], 
                       np.array(total_reward_means) - np.array(total_reward_stds),
                       np.array(total_reward_means) + np.array(total_reward_stds),
                       color=colors[2], alpha=0.15)
    
    ax.set_title('Tracking Rewards', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot learning metrics
    ax = fig.add_subplot(gs[0, 1])
    learning_metrics = ['train/value_loss', 'train/policy_loss', 'train/entropy_loss']
    print("\nLearning metrics:")
    colors = ['#2ca02c', '#d62728', '#9467bd']  # Green, red, purple
    for idx, metric in enumerate(learning_metrics):
        means = [aggregated_metrics[i][metric]['mean'] for i in iterations if metric in aggregated_metrics[i]]
        stds = [aggregated_metrics[i][metric]['std'] for i in iterations if metric in aggregated_metrics[i]]
        if means:
            print(f"{metric}: {len(means)} points, first value: {means[0]}")
            label = metric.replace('train/', '').replace('_loss', '')
            ax.plot(iterations[:len(means)], means, label=label, 
                   color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
            ax.fill_between(iterations[:len(means)], 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color=colors[idx], alpha=0.15)
    ax.set_title('Learning Metrics', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot component rewards
    ax = fig.add_subplot(gs[1, 0])
    component_metrics = [
        'train/rew_lin_vel_z', 'train/rew_base_height', 'train/rew_action_rate', 
        'train/rew_similar_to_default', 'train/rew_feet_air_time', 'train/rew_motion_tracking'
    ]
    print("\nComponent metrics:")
    colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']
    for idx, metric in enumerate(component_metrics):
        means = [aggregated_metrics[i][metric]['mean'] for i in iterations if metric in aggregated_metrics[i]]
        stds = [aggregated_metrics[i][metric]['std'] for i in iterations if metric in aggregated_metrics[i]]
        if means:
            print(f"{metric}: {len(means)} points, first value: {means[0]}")
            label = metric.replace('train/rew_', '').replace('_', ' ').title()
            ax.plot(iterations[:len(means)], means, label=label, 
                   color=colors[idx], linewidth=2.5, linestyle=line_styles[idx % len(line_styles)])
            ax.fill_between(iterations[:len(means)], 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color=colors[idx], alpha=0.15)
    ax.set_title('Component Rewards', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
                 frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot noise std and learning rate
    ax = fig.add_subplot(gs[1, 1])
    training_metrics = ['train/noise_std', 'train/learning_rate']
    print("\nTraining metrics:")
    colors = ['#ff9896', '#98df8a']  # Light red and light green
    for idx, metric in enumerate(training_metrics):
        means = [aggregated_metrics[i][metric]['mean'] for i in iterations if metric in aggregated_metrics[i]]
        stds = [aggregated_metrics[i][metric]['std'] for i in iterations if metric in aggregated_metrics[i]]
        if means:
            print(f"{metric}: {len(means)} points, first value: {means[0]}")
            label = metric.replace('train/', '').replace('_', ' ').title()
            ax.plot(iterations[:len(means)], means, label=label, 
                   color=colors[idx], linewidth=2.5, linestyle=line_styles[idx])
            ax.fill_between(iterations[:len(means)], 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color=colors[idx], alpha=0.15)
    ax.set_title('Training Parameters', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add overall title
    fig.suptitle('Training Progress', fontsize=16, y=0.95, fontweight='bold')
    
    # Calculate and print summary statistics
    if iterations:
        last_iter = max(iterations)
        print("\nSummary Statistics at Last Iteration:")
        print(f"Iteration: {last_iter}")
        
        # Calculate mean total reward (sum of tracking rewards)
        tracking_rewards = []
        for metric in total_reward_metrics:
            if metric in aggregated_metrics[last_iter]:
                tracking_rewards.append(aggregated_metrics[last_iter][metric]['mean'])
        mean_total_reward = sum(tracking_rewards) if tracking_rewards else 0
        print(f"Mean Total Reward: {mean_total_reward:.4f}")
        
        # Print individual tracking rewards
        for metric in total_reward_metrics:
            if metric in aggregated_metrics[last_iter]:
                print(f"{metric}: {aggregated_metrics[last_iter][metric]['mean']:.4f}")
        
        # Print learning metrics
        print("\nLearning Metrics:")
        for metric in learning_metrics:
            if metric in aggregated_metrics[last_iter]:
                print(f"{metric}: {aggregated_metrics[last_iter][metric]['mean']:.4f}")
        
        # Print component rewards
        print("\nComponent Rewards:")
        for metric in component_metrics:
            if metric in aggregated_metrics[last_iter]:
                print(f"{metric}: {aggregated_metrics[last_iter][metric]['mean']:.4f}")
    
    # Adjust layout and save
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved plot to {os.path.join(output_dir, 'training_metrics.png')}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the log directory containing training.txt")
    args = parser.parse_args()
    
    # Aggregate metrics from the log directory
    aggregated_metrics = aggregate_metrics(args.log_dir)
    
    # Plot the aggregated metrics
    plot_aggregated_metrics(aggregated_metrics, output_dir=args.log_dir, log_folder=args.log_dir)

if __name__ == "__main__":
    main()