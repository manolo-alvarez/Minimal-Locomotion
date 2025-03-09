"""ZBot training with TDMPC2

Run:
    python genesis_playground/zbot/zbot_tdmpc2_train.py --num_envs 64 --device mps
    (or cuda)
"""

import argparse
import os
import pickle
import wandb
import torch
import numpy as np
from datetime import datetime

import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Apply patches before importing TDMPC2
from patch_tdmpc2 import apply_patches
apply_patches()

# Now import the patched modules
from zbot_env import ZbotEnv
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.common.buffer import Buffer  # Make sure this file exists

import genesis as gs
from tensordict import TensorDict


class AttrDict(dict):
    """Dictionary that allows attribute-style access"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return AttrDict(super().copy())


def get_tdmpc2_cfg(log_dir, num_envs):
    """Configure TDMPC2 for ZBot training."""
    cfg = {
        # Environment
        "env_name": "ZBot",
        "obs_dim": 28,  # Will be overridden later
        "action_dim": 12,  # Will be overridden later
        "episode_length": 1000,
        
        # Q-function parameters (needed for TD-MPC2)
        "vmin": -10.0,  # Minimum Q-value for discretization
        "vmax": 10.0,   # Maximum Q-value for discretization
        
        # Training
        "seed": 1,
        "device": "mps",  # Use MPS on Apple Silicon
        "log_dir": log_dir,
        "save_interval": 50_000,
        "eval_interval": 10_000,
        "batch_size": 256,
        "max_steps": 100_000_000,
        "horizon": 3,
        "iterations": 6,
        
        # Architecture
        "latent_dim": 256,
        "hidden_dim": 512,
        "mlp_dim": 512,
        "num_layers": 4,
        "num_filters": 32,
        "num_q": 5,
        "task_dim": 96,
        "num_bins": 101,
        "dropout": 0.0,
        "log_std_min": -10.0,
        "log_std_max": 2.0,
        "simnorm_dim": 8,
        
        # Buffer
        "buffer_size": 1_000_000,
        "num_envs": num_envs,
        
        # Algorithm
        "lr": 3e-4,
        "enc_lr_scale": 1.0,
        "temperature": 10.0,
        "num_samples": 512,
        "num_elites": 64,
        "num_pi_trajs": 0,
        "min_std": 0.1,
        "max_std": 0.5,
        "tau": 0.005,
        "discount_min": 0.95,
        "discount_max": 0.99,
        "discount_denom": 200,
        "grad_clip_norm": 10.0,
        "consistency_coef": 1.0,
        "reward_coef": 1.0,
        "value_coef": 1.0,
        "rho": 0.5,
        "beta": 1.0,
        "discount": 0.99,
        "entropy_coef": 0.1,
        "mpc": True,
        
        # Misc
        "multitask": False,
        "tasks": [],
        "action_dims": [],
        "compile": False,
        "steps": 0,
        
        # Simplify observation setup
        "obs": "state",  # Not RGB
        "pixel_obs": False,  # Not using pixel observations
        "obs_shape": {"obs": None},  # Will be updated later
    }
    return cfg


def get_cfgs():
    env_cfg = {
        "num_actions": 10,
        # joint/link names
        # NOTE: hip roll/yaw flipped between sim & real robot FIXME
        "default_joint_angles": {  # [rad]
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
            "R_Hip_Pitch",
            "L_Hip_Pitch",
            "R_Hip_Yaw",
            "L_Hip_Yaw",
            "R_Hip_Roll",
            "L_Hip_Roll",
            "R_Knee_Pitch",
            "L_Knee_Pitch",
            "R_Ankle_Pitch",
            "L_Ankle_Pitch",
        ],
        # friction
        "env_friction_range": {
            "start": [0.9, 1.1],
            "end": [0.9, 1.1],
        },
        # link mass
        # varying this too much collapses the training
        "link_mass_multipliers": {
            "start": [1.0, 1.0],
            "end": [1.0, 1.0],
        },
        # RFI
        "rfi_scale": 0.1,
        # PD
        "kp": 20.0,
        "kd": 0.5,
        "kp_multipliers": [0.75, 1.25],
        "kd_multipliers": [0.75, 1.25],
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.41],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "max_torque": 10.0,
    }
    obs_cfg = {
        "num_obs": 39,
        # FIXME: IMU mounting orientation is different between sim & real robot
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        # Include names of observations to exclude from the actor observations
        "obs_exclusions": []
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "feet_air_time": 5.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_y_range": [0.0, 0.0],
        "lin_vel_x_range": [-0.2, 0.4],
        "ang_vel_range": [-0.4, 0.4],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


class WandbTDMPC2Logger:
    """Logger for TDMPC2 training with WandB integration."""
    
    def __init__(self, entity=None):
        self.entity = entity
        
    def log(self, info):
        # Convert any tensors to floats for wandb
        metrics = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                metrics[f'train/{k}'] = v.item()
            else:
                metrics[f'train/{k}'] = v
                
        wandb.log(metrics)


def create_episode_td(obs_list, action_list, reward_list, device):
    """Create a TensorDict for a complete episode."""
    # Ensure all lists have the same length
    assert len(obs_list) == len(action_list) == len(reward_list)
    
    # Stack tensors along time dimension
    obs_tensor = torch.stack(obs_list)       # [T, obs_dim]
    action_tensor = torch.stack(action_list) # [T, action_dim]
    reward_tensor = torch.stack(reward_list) # [T]
    
    # Create TensorDict with expected structure
    return TensorDict({
        'obs': obs_tensor,
        'action': action_tensor,
        'reward': reward_tensor,
    }, batch_size=[len(obs_list)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zbot-tdmpc2")
    parser.add_argument("-B", "--num_envs", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--show_viewer", type=bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--from_checkpoint", type=bool, default=False)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    
    gs.init(logging_level="warning")

    log_dir = f"{args.log_dir}/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Initialize WandB
    if args.use_wandb:
        wandb_run = wandb.init(
            project="zbot-tdmpc2",
            entity=args.wandb_entity,
            config={
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "command_cfg": command_cfg,
                "num_envs": args.num_envs,
                "max_steps": args.max_steps,
                "exp_name": args.exp_name,
            },
            name=args.exp_name,
        )
        logger = WandbTDMPC2Logger(entity=args.wandb_entity)
    else:
        logger = None
    
    # Create log directory
    if not args.from_checkpoint:
        if os.path.exists(log_dir):
            print(f"Removing existing log directory: {log_dir}")
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory: {log_dir}")
    
    # Create the environment
    env = ZbotEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )
    
    # Check MPS availability
    mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
    if mps_available and args.device == 'mps':
        device = 'mps'
        print("Using MPS device")
    elif torch.cuda.is_available() and args.device == 'cuda':
        device = 'cuda'
        print("Using CUDA device")
    else:
        device = 'cpu'
        print("Using CPU device")
    
    # Setup TDMPC2
    tdmpc2_cfg = get_tdmpc2_cfg(log_dir, args.num_envs)
    tdmpc2_cfg['device'] = device  # Use the device we determined
    tdmpc2_cfg['max_steps'] = args.max_steps
    
    # Fix: Use the correct attributes from ZbotEnv
    tdmpc2_cfg['obs_dim'] = env.num_obs
    tdmpc2_cfg['action_dim'] = env.num_actions
    
    # For state-based observations
    tdmpc2_cfg['pixel_obs'] = False
    
    # IMPORTANT: Create a valid obs_shape dictionary with 'obs' key
    # This matches what the encoder expects
    tdmpc2_cfg['obs_shape'] = {'obs': env.num_obs}
    
    # Add steps parameter required by Buffer
    tdmpc2_cfg['steps'] = args.max_steps * args.num_envs
    
    # Convert dict to AttrDict for attribute access
    tdmpc2_cfg = AttrDict(tdmpc2_cfg)
    
    # Create the TDMPC2 agent
    agent = TDMPC2(tdmpc2_cfg)
    
    # Initialize buffer
    buffer = Buffer(tdmpc2_cfg)
    
    # Load checkpoint if specified
    if args.from_checkpoint:
        checkpoint_path = f"{log_dir}/checkpoint.pt"
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    # Training loop
    print(f"Starting training for {args.max_steps} steps")
    obs = env.reset()
    
    # Process the observation
    if isinstance(obs, tuple):
        obs = obs[0]  # Get only the actor observations if it returns a tuple
    
    # Convert to tensor
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs).to(agent.device)
    
    # Print observation shape for debugging
    print(f"Initial observation shape: {obs.shape}")
    
    # Take a single environment's observation for evaluation and action selection
    # This is critical because TDMPC2 only supports one environment at a time for planning
    eval_obs = obs[0].unsqueeze(0) if obs.dim() > 1 else obs
    print(f"Evaluation observation shape: {eval_obs.shape}")
    
    # Initialize tracking variables
    done = torch.zeros(args.num_envs, dtype=torch.bool, device=agent.device)
    episode_reward = torch.zeros(args.num_envs, device=agent.device)
    episode_length = torch.zeros(args.num_envs, dtype=torch.long, device=agent.device)
    episode_rewards = []
    
    # Initialize episode storage as a list of lists (one list per environment)
    episode_obs = [[] for _ in range(args.num_envs)]
    episode_actions = [[] for _ in range(args.num_envs)]
    episode_rewards_buffer = [[] for _ in range(args.num_envs)]
    
    for step in range(args.max_steps):
        # Act in the environment using the evaluation observation
        with torch.no_grad():
            action = agent.act(eval_obs, t0=(episode_length[0] == 0), eval_mode=False)
            
            # Expand action to all environments
            if args.num_envs > 1:
                action_expanded = action.unsqueeze(0).expand(args.num_envs, -1)
                # IMPORTANT: Keep as tensor for environment step function
                action_tensor = action_expanded
            else:
                action_tensor = action.unsqueeze(0) if action.dim() == 1 else action
        
        # Store current observations and actions
        for env_idx in range(args.num_envs):
            episode_obs[env_idx].append(obs[env_idx].clone())
            episode_actions[env_idx].append(action_tensor[env_idx].clone())
        
        # Step the environment with tensor
        next_obs, reward, dones, info = env.step(action_tensor)
        
        # Convert next_obs to tensor if needed
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.FloatTensor(next_obs).to(agent.device)
        
        # Reshape if needed
        if next_obs.dim() == 1:
            next_obs = next_obs.unsqueeze(0)
            
        # Convert rewards and dones to tensors if needed
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward).to(agent.device)
        if isinstance(dones, np.ndarray):
            done = torch.FloatTensor(dones).to(agent.device).bool()
        else:
            done = dones.bool()
        
        # Store rewards AFTER getting them from env.step()
        for env_idx in range(args.num_envs):
            episode_rewards_buffer[env_idx].append(reward[env_idx].clone())
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        
        # Check for episode completion
        if done.any() or episode_length.max() >= tdmpc2_cfg['episode_length']:
            # Add completed episodes to buffer
            for env_idx in range(args.num_envs):
                if done[env_idx] or episode_length[env_idx] >= tdmpc2_cfg['episode_length']:
                    # Get episode data for this environment
                    ep_length = episode_length[env_idx].item()
                    
                    # Skip if episode is empty
                    if ep_length == 0:
                        print(f"Warning: Empty episode for environment {env_idx}. Skipping.")
                        continue
                    
                    try:
                        # Create TensorDict with episode data
                        td = TensorDict({
                            'obs': torch.stack(episode_obs[env_idx]),
                            'action': torch.stack(episode_actions[env_idx]), 
                            'reward': torch.stack(episode_rewards_buffer[env_idx]),
                        }, batch_size=[len(episode_obs[env_idx])])
                        
                        # Add task if multitask is enabled
                        if tdmpc2_cfg['multitask']:
                            td['task'] = torch.zeros(ep_length, dtype=torch.long, device=agent.device)
                        
                        # Add to buffer - print shape for debugging
                        buffer.add(td)
                        
                    except Exception as e:
                        print(f"Error creating TensorDict for environment {env_idx}: {e}")
                    
                    # Track statistics
                    if done[env_idx]:
                        episode_rewards.append(episode_reward[env_idx].item())
                        if len(episode_rewards) > 10:
                            episode_rewards.pop(0)
                    
                    # Reset metrics for this environment
                    episode_reward[env_idx] = 0
                    episode_length[env_idx] = 0
            
            # Reset episode storage for completed episodes
            for env_idx in range(args.num_envs):
                if done[env_idx] or episode_length[env_idx] >= tdmpc2_cfg['episode_length']:
                    episode_obs[env_idx] = []
                    episode_actions[env_idx] = []
                    episode_rewards_buffer[env_idx] = []
            
            # Reset environment
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(agent.device)
        else:
            # Continue episode
            obs = next_obs
        
        # Update model after we have enough data
        if buffer.num_eps > tdmpc2_cfg['batch_size'] and step % 10 == 0:
            update_info = agent.update(buffer)
            
            # Log to WandB
            if args.use_wandb and step % 100 == 0:
                update_info['avg_episode_reward'] = np.mean(episode_rewards) if episode_rewards else 0
                update_info['step'] = step
                logger.log(update_info)
        
        # Print progress
        if step % 1000 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Step: {step}, Avg Reward: {avg_reward:.2f}, Buffer Episodes: {buffer.num_eps}")
        
        # Save checkpoint
        if step > 0 and step % tdmpc2_cfg['save_interval'] == 0:
            checkpoint_path = f"{log_dir}/checkpoint_{step}.pt"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = f"{log_dir}/checkpoint_final.pt"
    agent.save(final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    
    if args.use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    main()