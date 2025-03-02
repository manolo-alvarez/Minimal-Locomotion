#!/usr/bin/env python3
"""
Visualize ZBot training progress from log files.
"""

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict

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

def aggregate_metrics(log_folder):
    """Aggregate metrics from multiple training runs."""
    # Find all training.txt files in subdirectories
    training_files = glob.glob(os.path.join(log_folder, '**/training.txt'), recursive=True)
    
    if not training_files:
        raise ValueError(f"No training.txt files found in {log_folder}")
    
    print(f"Found {len(training_files)} training runs to analyze")
    
    # Parse each training log
    all_metrics = []
    for log_file in training_files:
        try:
            metrics = parse_training_log(log_file)
            all_metrics.append(metrics)
            print(f"Processed {log_file}: {len(metrics['iterations'])} iterations")
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    if not all_metrics:
        raise ValueError("No valid training logs could be processed")
    
    # Find the maximum number of iterations across all runs
    max_iterations = max([max(m['iterations']) for m in all_metrics])
    
    # Create a common x-axis for all runs
    common_iterations = list(range(1, max_iterations + 1))
    
    # Initialize aggregated metrics
    aggregated = {
        'iterations': common_iterations,
        'metrics': defaultdict(lambda: {'values': [], 'mean': [], 'min': [], 'max': []})
    }
    
    # List of metric names (excluding iterations)
    metric_names = [
        'total_rewards', 'episode_lengths', 'tracking_lin_vel', 'tracking_ang_vel',
        'lin_vel_z', 'base_height', 'action_rate', 'similar_to_default', 'feet_air_time'
    ]
    
    # For each iteration, collect values from all runs
    for iteration in common_iterations:
        for metric_name in metric_names:
            values_at_iteration = []
            
            for run_metrics in all_metrics:
                # Find the value at this iteration if it exists
                if iteration in run_metrics['iterations']:
                    idx = run_metrics['iterations'].index(iteration)
                    values_at_iteration.append(run_metrics[metric_name][idx])
            
            if values_at_iteration:
                aggregated['metrics'][metric_name]['values'].append(values_at_iteration)
                aggregated['metrics'][metric_name]['mean'].append(np.mean(values_at_iteration))
                aggregated['metrics'][metric_name]['min'].append(np.min(values_at_iteration))
                aggregated['metrics'][metric_name]['max'].append(np.max(values_at_iteration))
            else:
                # No data for this iteration
                aggregated['metrics'][metric_name]['values'].append([])
                aggregated['metrics'][metric_name]['mean'].append(None)
                aggregated['metrics'][metric_name]['min'].append(None)
                aggregated['metrics'][metric_name]['max'].append(None)
    
    return aggregated

def plot_aggregated_metrics(aggregated_metrics, output_dir=None):
    """Plot the aggregated metrics with variance."""
    iterations = aggregated_metrics['iterations']
    
    plt.figure(figsize=(12, 8))
    
    # Plot total reward
    plt.subplot(2, 2, 1)
    metric = aggregated_metrics['metrics']['total_rewards']
    mean_values = np.array(metric['mean'])
    min_values = np.array(metric['min'])
    max_values = np.array(metric['max'])
    
    # Filter out None values
    valid_indices = ~np.isnan(mean_values)
    valid_iterations = np.array(iterations)[valid_indices]
    valid_mean = mean_values[valid_indices]
    valid_min = min_values[valid_indices]
    valid_max = max_values[valid_indices]
    
    plt.plot(valid_iterations, valid_mean, 'b-', linewidth=2, label='Mean')
    plt.fill_between(valid_iterations, valid_min, valid_max, color='b', alpha=0.2, label='Range')
    plt.title('Mean Total Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot episode length
    plt.subplot(2, 2, 2)
    metric = aggregated_metrics['metrics']['episode_lengths']
    mean_values = np.array(metric['mean'])
    min_values = np.array(metric['min'])
    max_values = np.array(metric['max'])
    
    # Filter out None values
    valid_indices = ~np.isnan(mean_values)
    valid_iterations = np.array(iterations)[valid_indices]
    valid_mean = mean_values[valid_indices]
    valid_min = min_values[valid_indices]
    valid_max = max_values[valid_indices]
    
    plt.plot(valid_iterations, valid_mean, 'g-', linewidth=2, label='Mean')
    plt.fill_between(valid_iterations, valid_min, valid_max, color='g', alpha=0.2, label='Range')
    plt.title('Mean Episode Length')
    plt.xlabel('Iteration')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.legend()
    
    # Plot tracking rewards
    plt.subplot(2, 2, 3)
    
    # Linear velocity tracking
    metric = aggregated_metrics['metrics']['tracking_lin_vel']
    mean_values = np.array(metric['mean'])
    min_values = np.array(metric['min'])
    max_values = np.array(metric['max'])
    
    valid_indices = ~np.isnan(mean_values)
    valid_iterations = np.array(iterations)[valid_indices]
    valid_mean = mean_values[valid_indices]
    valid_min = min_values[valid_indices]
    valid_max = max_values[valid_indices]
    
    plt.plot(valid_iterations, valid_mean, 'r-', linewidth=2, label='Linear Velocity')
    plt.fill_between(valid_iterations, valid_min, valid_max, color='r', alpha=0.2)
    
    # Angular velocity tracking
    metric = aggregated_metrics['metrics']['tracking_ang_vel']
    mean_values = np.array(metric['mean'])
    min_values = np.array(metric['min'])
    max_values = np.array(metric['max'])
    
    valid_indices = ~np.isnan(mean_values)
    valid_iterations = np.array(iterations)[valid_indices]
    valid_mean = mean_values[valid_indices]
    valid_min = min_values[valid_indices]
    valid_max = max_values[valid_indices]
    
    plt.plot(valid_iterations, valid_mean, 'm-', linewidth=2, label='Angular Velocity')
    plt.fill_between(valid_iterations, valid_min, valid_max, color='m', alpha=0.2)
    
    plt.title('Tracking Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot penalty rewards
    plt.subplot(2, 2, 4)
    
    # Define colors for each metric
    colors = {
        'lin_vel_z': 'c',
        'base_height': 'y',
        'action_rate': 'k',
        'similar_to_default': 'b',
        'feet_air_time': 'g'
    }
    
    # Define line styles
    styles = {
        'lin_vel_z': '-',
        'base_height': '-',
        'action_rate': '-',
        'similar_to_default': '--',
        'feet_air_time': '--'
    }
    
    # Plot each component reward
    for metric_name, color in colors.items():
        metric = aggregated_metrics['metrics'][metric_name]
        mean_values = np.array(metric['mean'])
        min_values = np.array(metric['min'])
        max_values = np.array(metric['max'])
        
        valid_indices = ~np.isnan(mean_values)
        valid_iterations = np.array(iterations)[valid_indices]
        valid_mean = mean_values[valid_indices]
        valid_min = min_values[valid_indices]
        valid_max = max_values[valid_indices]
        
        if len(valid_iterations) > 0:
            plt.plot(valid_iterations, valid_mean, color + styles[metric_name], linewidth=2, label=metric_name.replace('_', ' ').title())
            plt.fill_between(valid_iterations, valid_min, valid_max, color=color, alpha=0.1)
    
    plt.title('Component Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'aggregated_training_progress.png'), dpi=300)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize ZBot training progress')
    parser.add_argument('--log_folder', type=str, default='results/zbot-walking',
                        help='Path to the folder containing training runs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots')
    args = parser.parse_args()
    
    # Aggregate metrics from all training runs
    aggregated_metrics = aggregate_metrics(args.log_folder)
    
    # Plot the aggregated metrics
    plot_aggregated_metrics(aggregated_metrics, args.output_dir)
    
    # Print summary statistics
    total_runs = len(glob.glob(os.path.join(args.log_folder, '**/training.txt'), recursive=True))
    print(f"Processed {total_runs} training runs")
    
    # Find the last valid iteration with data
    last_valid_iteration = 0
    for i in range(len(aggregated_metrics['iterations'])-1, -1, -1):
        if aggregated_metrics['metrics']['total_rewards']['mean'][i] is not None:
            last_valid_iteration = aggregated_metrics['iterations'][i]
            break
    
    if last_valid_iteration > 0:
        idx = aggregated_metrics['iterations'].index(last_valid_iteration)
        print(f"Statistics at iteration {last_valid_iteration}:")
        print(f"  Mean total reward: {aggregated_metrics['metrics']['total_rewards']['mean'][idx]:.4f}")
        print(f"  Mean episode length: {aggregated_metrics['metrics']['episode_lengths']['mean'][idx]:.2f}")
        print(f"  Reward range: [{aggregated_metrics['metrics']['total_rewards']['min'][idx]:.4f}, {aggregated_metrics['metrics']['total_rewards']['max'][idx]:.4f}]")

if __name__ == "__main__":
    main() 