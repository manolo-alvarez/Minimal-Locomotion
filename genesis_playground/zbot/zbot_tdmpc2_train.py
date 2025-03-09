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


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
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
            "class_name": "ActorCritic",
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 48,
            "policy_class_name": "ActorCritic",
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
        "empirical_normalization": False
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

def get_tdmpc2_cfg(env_cfg, obs_cfg, reward_cfg, command_cfg, log_dir, exp_name="zbot-tdmpc2"):
    # Basic TDMPC2 configuration
    cfg = {
        # Model size - choose based on your compute resources
        "model_size": 5,  # Can be one of [1, 5, 19, 48, 317]
        
        # Environment settings (from your env config)
        "episode_length": int(env_cfg["episode_length_s"] * 50),  # Assuming 50Hz control frequency
        "action_dim": env_cfg["num_actions"],
        "obs_dim": obs_cfg["num_obs"],
        
        # Training parameters
        "steps": env_cfg["episode_length_s"] * 50 * 1000,  # Total training steps
        "horizon": 8,  # Planning horizon
        "batch_size": 256,
        "lr": 1e-4,
        "discount_min": 0.95,
        "discount_max": 0.99,
        "discount_denom": 200,
        
        # MPC parameters
        "mpc": True,
        "iterations": 5,
        "num_samples": 512,
        "num_elites": 64,
        "num_pi_trajs": 8,
        "temperature": 0.5,
        "min_std": 0.05,
        "max_std": 0.5,
        
        # Device
        "device": "mps" if torch.mps.is_available() else "cpu",
        "compile": False,  # Set to True if using newer PyTorch versions for speedup
        
        # Task-related
        "multitask": False,
        
        # Loss coefficients
        "consistency_coef": 1.0,
        "reward_coef": 1.0,
        "value_coef": 1.0,
        "rho": 0.5,
        "grad_clip_norm": 10.0,
        "num_q": 2,
        
        # Buffer settings
        "capacity": 1_000_000,
        "num_workers": 4,
        
        # Logging and saving
        "work_dir": os.path.join(log_dir, exp_name),
        "checkpoint": None,
        "seed": 1,
    }
    
    return cfg

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

class ZbotTDMPC2Wrapper:
    def __init__(self, zbot_env):
        self.env = zbot_env
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self._current_obs = None
        
    def _get_observation_space(self):
        # Create a gym-like observation space
        import gym
        import numpy as np
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.env.obs_cfg["num_obs"],)
        )
    
    def _get_action_space(self):
        # Create a gym-like action space
        import gym
        import numpy as np
        return gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.env.env_cfg["num_actions"],)
        )
        
    def reset(self, task_idx=None):
        # Reset the environment and return the initial observation
        self.env.reset()
        self._current_obs = self.env.obs_buf[0].clone().cpu().numpy()
        return self._current_obs
        
    def step(self, action):
        # Convert action to tensor if necessary
        action_tensor = torch.from_numpy(action).to(self.env.device).unsqueeze(0)
        
        # Apply action and get rewards, next observations
        self.env.step(action_tensor)
        
        # Get the observation for the first environment (for single-env case)
        obs = self.env.obs_buf[0].clone().cpu().numpy()
        reward = self.env.rew_buf[0].item()
        done = self.env.reset_buf[0].item() > 0
        
        # Save current observation
        self._current_obs = obs
        
        # Return step results with an empty info dict
        return obs, reward, done, {}
        
    def render(self):
        # Return an RGB array for visualization if needed
        # This might not be directly available from Genesis
        return None

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
    
    # Import torch here to avoid potential circular imports
    import torch
    
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
    
    # Wrap it for TDMPC2
    wrapped_env = ZbotTDMPC2Wrapper(zbot_env)
    
    # Get TDMPC2 configuration
    tdmpc2_cfg = get_tdmpc2_cfg(env_cfg, obs_cfg, reward_cfg, command_cfg, log_dir)
    
    # Import TDMPC2 components
    from tdmpc2 import TDMPC2
    from tdmpc2.common.buffer import Buffer
    from tdmpc2.trainer.online_trainer import OnlineTrainer
    from tdmpc2.common.logger import Logger
    
    # Setup TDMPC2
    agent = TDMPC2(tdmpc2_cfg)
    buffer = Buffer(tdmpc2_cfg)
    logger = Logger(tdmpc2_cfg)
    
    # Create WandbLogger wrapper if using wandb
    if args.use_wandb:
        class WandbLogger:
            def __init__(self, original_logger):
                self.logger = original_logger
                
            def log(self, info):
                # Log with original logger
                self.logger.log(info)
                
                # Log to wandb
                wandb_metrics = {}
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        wandb_metrics[k] = v
                wandb.log(wandb_metrics)
        
        wandb.init(
            project=args.exp_name,
            entity=args.wandb_entity,
            name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "command_cfg": command_cfg,
                "tdmpc2_cfg": tdmpc2_cfg
            }
        )
        
        logger = WandbLogger(logger)
    
    # Initialize trainer
    trainer = OnlineTrainer(
        cfg=tdmpc2_cfg,
        env=wrapped_env,
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
