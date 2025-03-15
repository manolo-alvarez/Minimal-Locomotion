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

from zbot_env_v2 import ZbotEnv2
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    
    # I think the genesis_playground was used to an older version of rsl_rl
    # i had to modify the cfg a little to get the latest rsl_rl version to work

    # most of these changes are moving existing config to other places in the dict,
    # and adding "class_name" which are just labels. 
    # I did add "empirical_normalization": False becuase the rsl_rl library requires it in 
    # the config, and having it off seems like the intended behavior for this training script. 
    # Not sure what it does, maybe we can leverage it in the future.
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


def get_cfgs():
    env_cfg = {
        "num_actions": 10,
        # joint/link names
        # NOTE: hip roll/yaw flipped between sim & real robot FIXME
        "default_joint_angles": {  # [rad]
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
            "left_ankle",
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

class WandbOnPolicyRunner(OnPolicyRunner):
    def log(self, info):
        super().log(info)
        # Extract reward values from each episode info
        if len(info['ep_infos']) > 0:  # Check if there are any episodes
            metrics = {
                'train/rew_tracking_lin_vel': [ep_info['rew_tracking_lin_vel'] for ep_info in info['ep_infos']],
                'train/rew_tracking_ang_vel': [ep_info['rew_tracking_ang_vel'] for ep_info in info['ep_infos']],
                'train/rew_lin_vel_z': [ep_info['rew_lin_vel_z'] for ep_info in info['ep_infos']],
                'train/rew_base_height': [ep_info['rew_base_height'] for ep_info in info['ep_infos']],
                'train/rew_action_rate': [ep_info['rew_action_rate'] for ep_info in info['ep_infos']],
                'train/rew_similar_to_default': [ep_info['rew_similar_to_default'] for ep_info in info['ep_infos']],
                'train/rew_feet_air_time': [ep_info['rew_feet_air_time'] for ep_info in info['ep_infos']],
                'train/full_reward': [info['rewards'].mean()],
                'train/mean_value_loss': [info['mean_value_loss']],
                'train/mean_surrogate_loss': [info['mean_surrogate_loss']],
                'train/mean_entropy': [info['mean_entropy']]
            }
            
            # For wandb, you might want to average the values
            avg_metrics = {k: sum(v)/len(v) if v else 0 for k, v in metrics.items()}
            wandb.log(avg_metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zbot-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
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
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if not args.from_checkpoint:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    env = ZbotEnv2(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        device=args.device,
        show_viewer=args.show_viewer,
    )

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )
    
    if args.use_wandb:
        run = wandb.init(
                project="PPO",
                entity=args.wandb_entity,
                name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "num_envs": args.num_envs,
                    "max_iterations": args.max_iterations,
                    "device": args.device,
                    "env_cfg": env_cfg,
                    "obs_cfg": obs_cfg,
                    "reward_cfg": reward_cfg,
                    "command_cfg": command_cfg,
                    "train_cfg": train_cfg,
                }
        )
        runner = WandbOnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
