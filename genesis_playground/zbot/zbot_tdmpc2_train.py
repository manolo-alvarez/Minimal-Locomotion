""" ZBot training

Run:
    python genesis_playground/zbot/zbot_train.py --num_envs 4096 --max_iterations 200 --device mps 
    (or cuda)
"""

import argparse
import os
import pickle
import shutil
import wandb
from datetime import datetime

from zbot_env import ZbotEnv

import genesis as gs


import sys
import os

from tdmpc2.trainer.online_trainer import OnlineTrainer
import torch
import numpy as np
from tdmpc2 import TDMPC2
from tdmpc2.common.buffer import Buffer
from tdmpc2.common.logger import Logger

def get_train_cfg(exp_name, max_iterations, device="mps"):
    """
    Create a configuration dictionary that matches the structure from config.yaml
    but keeps some parameters from the original train_cfg
    """
    # Create a config that matches the YAML format
    train_cfg_dict = {
        # environment
        "task": "zbot",
        "task_title": "ZBot Locomotion",
        "obs": "state",
        
        # evaluation
        "checkpoint": None,
        "eval_episodes": 10,
        "eval_freq": 50000,
        
        # training
        "steps": max_iterations * 48 * 100,  # Convert iterations to steps
        "batch_size": 256,
        "reward_coef": 0.1,
        "value_coef": 0.1,
        "consistency_coef": 20,
        "rho": 0.5,
        "lr": 3e-4,
        "enc_lr_scale": 0.3,
        "grad_clip_norm": 20,
        "tau": 0.01,
        "discount_denom": 5,
        "discount_min": 0.95,
        "discount_max": 0.995,
        "buffer_size": 1_000_000,
        "capacity": 1_000_000,  # Alias for buffer_size used in TDMPC2
        "exp_name": exp_name,
        
        # planning
        "mpc": True,
        "iterations": 6,
        "num_samples": 512,
        "num_elites": 64,
        "num_pi_trajs": 24,
        "horizon": 3,
        "min_std": 0.05,
        "max_std": 2,
        "temperature": 0.5,
        
        # actor
        "log_std_min": -10,
        "log_std_max": 2,
        "entropy_coef": 1e-4,
        
        # critic
        "num_bins": 101,
        "vmin": -10,
        "vmax": 10,
        
        # architecture
        "model_size": 5,  # Choose from [1, 5, 19, 48, 317]
        "num_enc_layers": 2,
        "enc_dim": 256,
        "num_channels": 32,
        "mlp_dim": 512,
        "latent_dim": 512,
        "task_dim": 0,
        "num_q": 5,
        "dropout": 0.01,
        "simnorm_dim": 8,
        
        # logging
        "wandb_project": exp_name,
        "wandb_entity": None,
        "wandb_silent": False,
        "enable_wandb": False,  # We'll handle wandb separately
        "save_csv": True,
        
        # misc
        "save_video": True,
        "save_agent": True,
        "seed": 1,
        "device": device,
        "compile": False,
        
        # essential parameters for TDMPC2 that aren't in the YAML
        "work_dir": f"logs/{exp_name}",
        "multitask": False,
        "num_workers": 4,  # For data loading
        
        # These will be filled in by the environment wrapper
        "obs_shape": {"observations": 39},  # Will be overridden
        "obs_dim": 39,  # Will be overridden
        "action_dim": 10,  # Will be overridden
        "episode_length": 1000,  # Will be overridden
    }

    return train_cfg_dict


def get_env_cfg():
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

class WandbOnPolicyRunner(OnlineTrainer):
    def log(self, info):
        super().log(info)
        # Log metrics to wandb
        metrics = {
            'train/rew_tracking_lin_vel': info['train/rew_tracking_lin_vel'],
            'train/rew_tracking_ang_vel': info['train/rew_tracking_ang_vel'],
            'train/rew_lin_vel_z': info['train/rew_lin_vel_z'],
            'train/rew_base_height': info['train/rew_base_height'],
            'train/rew_action_rate': info['train/rew_action_rate'],
            'train/rew_similar_to_default': info['train/rew_similar_to_default'],
        }
        wandb.log(metrics)

class DotDict(dict):
    """
    A dictionary that allows attribute-style access
    while still behaving like a regular dictionary.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dictionaries to DotDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
        
    # Optional: Convert nested dictionaries when setting items
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super().__setitem__(key, value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zbot-tdmpc2")
    parser.add_argument("-B", "--num_envs", type=int, default=1)  # TDMPC2 works with single env
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--show_viewer", type=bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--from_checkpoint", type=bool, default=False)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    
    gs.init(logging_level="warning")

    log_dir = f"{args.log_dir}/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_env_cfg()
    
    if not args.from_checkpoint:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    # Create Genesis ZBot environment
    zbot_env = ZbotEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        device=args.device,
        show_viewer=args.show_viewer,
    )

    # Create from existing dictionary
    tdmpc2_cfg = DotDict(get_train_cfg(args.exp_name, args.max_iterations, device=args.device))
    tdmpc2_cfg_dict = get_train_cfg(args.exp_name, args.max_iterations, device=args.device)
    
    # Update config with environment-specific values
    tdmpc2_cfg.obs_shape = {"state": obs_cfg["num_obs"]}
    tdmpc2_cfg.obs_dim = obs_cfg["num_obs"]
    tdmpc2_cfg.action_dim = env_cfg["num_actions"]
    tdmpc2_cfg.episode_length = int(env_cfg["episode_length_s"] * 50)  # Assuming 50Hz control
    tdmpc2_cfg.work_dir = log_dir
    
    # Add wandb config if using wandb
    if args.use_wandb:
        tdmpc2_cfg.wandb_entity = args.wandb_entity
        tdmpc2_cfg.enable_wandb = True
    
    # Setup TDMPC2
    agent = TDMPC2(tdmpc2_cfg)
    buffer = Buffer(tdmpc2_cfg)
    logger = Logger(tdmpc2_cfg, tdmpc2_cfg_dict)
    
    # Create WandbLogger wrapper if using wandb
    #if args.use_wandb:
    #    class WandbLogger:
    #        def __init__(self, original_logger):
    #            self.logger = original_logger
    #            
    #        def log(self, info):
    #            # Log with original logger
    #            self.logger.log(info)
    #            
    #            # Log to wandb
    #            wandb_metrics = {}
    #            for k, v in info.items():
    #                if isinstance(v, (int, float)):
    #                    wandb_metrics[k] = v
    #            wandb.log(wandb_metrics)
    #    
    #    wandb.init(
    #        project=args.exp_name,
    #        entity=args.wandb_entity,
    #        name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #        config={
    #            "env_cfg": env_cfg,
    #            "obs_cfg": obs_cfg,
    #            "reward_cfg": reward_cfg,
    #            "command_cfg": command_cfg,
    #            "tdmpc2_cfg": tdmpc2_cfg
    #        }
    #    )
    #    
    #    logger = WandbLogger(logger)
    
    # Initialize trainer
    trainer = OnlineTrainer(
        cfg=tdmpc2_cfg,
        env=zbot_env,
        agent=agent,
        buffer=buffer,
        logger=logger
    )
    
    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, tdmpc2_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    # Train agent
    try:
        trainer.train()
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
