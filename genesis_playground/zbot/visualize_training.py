#!/usr/bin/env python3
"""
Visualize ZBot training progress from log files.
"""

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_training_log(log_file):
    """Parse the training log file to extract metrics."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract iterations and metrics using regex
    iterations = []
    total_rewards = []
    episode_lengths = []
    tracking_lin_vel = []
    tracking_ang_vel = []
    lin_vel_z = []
    base_height = []
    action_rate = []
    similar_to_default = []
    feet_air_time = []
    
    # Pattern to match iteration blocks
    iteration_pattern = r"Learning iteration (\d+)/\d+.*?Mean total reward: ([\d\.\-]+).*?Mean episode length: ([\d\.\-]+).*?Mean episode rew_tracking_lin_vel: ([\d\.\-]+).*?Mean episode rew_tracking_ang_vel: ([\d\.\-]+).*?Mean episode rew_lin_vel_z: ([\d\.\-]+).*?Mean episode rew_base_height: ([\d\.\-]+).*?Mean episode rew_action_rate: ([\d\.\-]+).*?Mean episode rew_similar_to_default: ([\d\.\-]+).*?Mean episode rew_feet_air_time: ([\d\.\-]+)"
    
    matches = re.finditer(iteration_pattern, content, re.DOTALL)
    
    for match in matches:
        iterations.append(int(match.group(1)))
        total_rewards.append(float(match.group(2)))
        episode_lengths.append(float(match.group(3)))
        tracking_lin_vel.append(float(match.group(4)))
        tracking_ang_vel.append(float(match.group(5)))
        lin_vel_z.append(float(match.group(6)))
        base_height.append(float(match.group(7)))
        action_rate.append(float(match.group(8)))
        similar_to_default.append(float(match.group(9)))
        feet_air_time.append(float(match.group(10)))
    
    return {
        'iterations': iterations,
        'total_rewards': total_rewards,
        'episode_lengths': episode_lengths,
        'tracking_lin_vel': tracking_lin_vel,
        'tracking_ang_vel': tracking_ang_vel,
        'lin_vel_z': lin_vel_z,
        'base_height': base_height,
        'action_rate': action_rate,
        'similar_to_default': similar_to_default,
        'feet_air_time': feet_air_time
    }

def plot_metrics(metrics, output_dir=None):
    """Plot the extracted metrics."""
    plt.figure(figsize=(12, 8))
    
    # Plot total reward
    plt.subplot(2, 2, 1)
    plt.plot(metrics['iterations'], metrics['total_rewards'], 'b-', linewidth=2)
    plt.title('Mean Total Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot episode length
    plt.subplot(2, 2, 2)
    plt.plot(metrics['iterations'], metrics['episode_lengths'], 'g-', linewidth=2)
    plt.title('Mean Episode Length')
    plt.xlabel('Iteration')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot tracking rewards
    plt.subplot(2, 2, 3)
    plt.plot(metrics['iterations'], metrics['tracking_lin_vel'], 'r-', label='Linear Velocity')
    plt.plot(metrics['iterations'], metrics['tracking_ang_vel'], 'm-', label='Angular Velocity')
    plt.title('Tracking Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot penalty rewards
    plt.subplot(2, 2, 4)
    plt.plot(metrics['iterations'], metrics['lin_vel_z'], 'c-', label='Lin Vel Z')
    plt.plot(metrics['iterations'], metrics['base_height'], 'y-', label='Base Height')
    plt.plot(metrics['iterations'], metrics['action_rate'], 'k-', label='Action Rate')
    plt.plot(metrics['iterations'], metrics['similar_to_default'], 'b--', label='Similar to Default')
    plt.plot(metrics['iterations'], metrics['feet_air_time'], 'g--', label='Feet Air Time')
    plt.title('Component Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=300)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize ZBot training progress')
    parser.add_argument('--log_file', type=str, default='results/zbot-walking/training.txt',
                        help='Path to the training log file')
    parser.add_argument('--output_dir', type=str, default='results/zbot-walking/plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    metrics = parse_training_log(args.log_file)
    plot_metrics(metrics, args.output_dir)
    
    print(f"Processed {len(metrics['iterations'])} training iterations")
    print(f"Final mean reward: {metrics['total_rewards'][-1]:.4f}")
    print(f"Final mean episode length: {metrics['episode_lengths'][-1]:.2f}")

if __name__ == "__main__":
    main() 