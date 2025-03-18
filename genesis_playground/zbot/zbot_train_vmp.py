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
import statistics

# Make VAEConditionedActorCritic available globally
globals()['VAEConditionedActorCritic'] = VAEConditionedActorCritic

def get_train_cfg(exp_name, max_iterations, latent_dim):
    """ Unified training config including original and VMP parameters """
    return {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.02,
            "entropy_coef": 0.1,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 6,
            "num_mini_batches": 6,
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
            "init_noise_std": 0.1,
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
            # "R_Hip_Pitch": 0.0,
            # "L_Hip_Pitch": 0.0,
            # "R_Hip_Yaw": 0.0,
            # "L_Hip_Yaw": 0.0,
            # "R_Hip_Roll": 0.0,
            # "L_Hip_Roll": 0.0,
            # "R_Knee_Pitch": 0.0,
            # "L_Knee_Pitch": 0.0,
            # "R_Ankle_Pitch": 0.0,
            # "L_Ankle_Pitch": 0.0,
            "right_hip_pitch": 0.0,
            "left_hip_pitch": 0.0,
            "right_hip_yaw": 0.0,
            "left_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "left_hip_roll": 0.0,
            "right_knee": 0.0,
            "left_knee": 0.0,
            "right_ankle": 0.0,
            "left_ankle": 0.0,
        },
        # "dof_names": [
        #     "R_Hip_Pitch", "L_Hip_Pitch", "R_Hip_Yaw", "L_Hip_Yaw",
        #     "R_Hip_Roll", "L_Hip_Roll", "R_Knee_Pitch", "L_Knee_Pitch",
        #     "R_Ankle_Pitch", "L_Ankle_Pitch",
        # ],
        "dof_names": [
            "right_hip_pitch",
            "left_hip_pitch",
            "right_hip_yaw",
            "left_hip_yaw",
            "right_hip_roll",
            "left_hip_roll",
            "right_knee",
            "left_knee",
            "right_ankle",
            "left_ankle",],
            
        
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
        "episode_length_s": 10.0,  # Reduced from 20.0 to 2.0 seconds for testing
        "resampling_time_s": 4.0,
        "action_scale": 0.5,
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
            "lin_vel": 50.0,
            "ang_vel": 0.25,
            "dof_pos": 10.0,
            "dof_vel": 0.5,
            "motion": 0.5,  # VMP addition
            "latent": 2.0   # VMP addition
        },
        "obs_exclusions": []
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.0075,
        "reward_scales": {
            # Tracking rewards (r_track)
            "motion_tracking": 5.0,          # Main tracking component
            "tracking_lin_vel": 2.0,         # Part of tracking
            "tracking_ang_vel": 0.5,         # Part of tracking
            "base_height": -10.0,           # Part of tracking (height maintenance)
            
            # Smoothness rewards (r_smooth)
            "action_rate": -0.1,            # First-order smoothness
            "action_rate2": -0.05,          # Second-order smoothness (new)
            "torque_penalty": -0.01,        # Torque penalty (new)
            
            # Alive rewards
            "alive_bonus": 10,             # Basic survival reward
            "early_termination": -2.0       # Penalty for early termination
        },
        
        # Tracking weights (following paper's c_h, c_θ, c_v, etc.)
        "pos_weight": 1.0,                  # Position tracking (c_p)
        "rot_weight": 0.5,                  # Rotation tracking (c_θ)
        "vel_weight": 0.2,                  # Velocity tracking (c_v)
        "joint_weight": 0.5,                # Joint position tracking (c_q)
        "joint_vel_weight": 0.2,            # Joint velocity tracking (c_q˙)
        
        # Early termination thresholds
        "max_deviation_threshold": 1,     # Maximum allowed end-effector deviation
        "deviation_frames_threshold": 10     # Number of frames before termination
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
            noise_std_type="log",
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

        # Assuming you have a configuration section for hyperparameters
        # Adjust action noise standard deviation
        self.action_noise_std = 1  # Example value, adjust as needed

        # Adjust learning rate in the optimizer
        self.optimizer = torch.optim.Adam(self.alg.actor_critic.parameters(), lr=0.0001)  # Example value, adjust as needed

    def log(self, info):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        super().log(info)
        
        # Write to training log file
        if self.train_log_file:
            log_str = f"Iteration: {self.current_learning_iteration}\n"
            
            # Log losses
            log_str += f"train/value_loss: {info['mean_value_loss']:.4f}\n"
            log_str += f"train/policy_loss: {info['mean_surrogate_loss']:.4f}\n"
            log_str += f"train/entropy_loss: {info['mean_entropy']:.4f}\n"
            
            # Log episode info
            if 'ep_infos' in info and info['ep_infos']:
                for key in info['ep_infos'][0]:
                    if key not in info['ep_infos'][0]:
                        continue
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in info['ep_infos']:
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    log_str += f"train/{key}: {value:.4f}\n"
            
            # Log rewards and episode length
            if 'rewbuffer' in info and len(info['rewbuffer']) > 0:
                log_str += f"train/total_reward: {statistics.mean(info['rewbuffer']):.4f}\n"
                log_str += f"train/episode_length: {statistics.mean(info['lenbuffer']):.4f}\n"
            
            # Log noise std
            mean_std = self.alg.actor_critic.action_std.mean()
            log_str += f"train/noise_std: {mean_std.item():.4f}\n"
            
            # Log learning rate
            log_str += f"train/learning_rate: {self.alg.learning_rate:.4f}\n"
            
            # Add separator
            log_str += "-" * 80 + "\n"
            
            # Write and flush
            self.train_log_file.write(log_str)
            self.train_log_file.flush()
            
        # Log to wandb if enabled
        if hasattr(self, 'wandb'):
            metrics = {
                'train/value_loss': info['mean_value_loss'],
                'train/policy_loss': info['mean_surrogate_loss'],
                'train/entropy_loss': info['mean_entropy'],
                'train/noise_std': self.alg.actor_critic.action_std.mean().item(),
                'train/learning_rate': self.alg.learning_rate
            }
            
            # Add episode info
            if 'ep_infos' in info and info['ep_infos']:
                for key in info['ep_infos'][0]:
                    if key not in info['ep_infos'][0]:
                        continue
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in info['ep_infos']:
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    metrics[f'train/{key}'] = value.item()
            
            # Add rewards and episode length
            if 'rewbuffer' in info and len(info['rewbuffer']) > 0:
                metrics['train/total_reward'] = statistics.mean(info['rewbuffer'])
                metrics['train/episode_length'] = statistics.mean(info['lenbuffer'])
            
            wandb.log(metrics)

    def __del__(self):
        # Close the training log file when the runner is destroyed
        if hasattr(self, 'train_log_file') and self.train_log_file:
            self.train_log_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vae_model", type=str, required=True)
    parser.add_argument("--show_viewer", type=bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--log_dir", type=str, default="vmp_logs")
    args = parser.parse_args()
    
    gs.init(logging_level="warning")
    # Initialize environment and configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs(args.vae_model)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, latent_dim=64)

    # Create log directory
    log_dir = f"{args.log_dir}/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = ZbotEnvVMP(
        num_envs=args.num_envs, 
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        vae_model_path = args.vae_model,
        device=args.device,
        show_viewer=args.show_viewer,
    )

    # Save configs..
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Initialize WandB
    if args.use_wandb:
        run = wandb.init(
            project="zbot-vmp",
            entity=args.wandb_entity,
            name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations,
                "device": args.device,
                "vae_model": args.vae_model,
                **env_cfg, **obs_cfg, **reward_cfg, **command_cfg
            }
        )
        runner = CustomOnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    else:
        runner = CustomOnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        if args.use_wandb:
            wandb.finish()
        # Save final model
        torch.save({
            "policy": runner.alg.actor_critic.state_dict(),
            "vae": env.vae.state_dict()
        }, os.path.join(log_dir, "final_model.pth"))

if __name__ == "__main__":
    main()