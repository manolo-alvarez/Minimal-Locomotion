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
import seaborn as sns

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
    print("Aggregating metrics from: ", log_folder)
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

def plot_aggregated_metrics(aggregated_metrics, output_dir=None, log_folder=None):
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
        # Generate filename based on log_folder
        if log_folder:
            # Extract the last part of the path to use as filename
            print("Saving plot as: ", os.path.join(output_dir, log_folder) + "_training_progress.png")
            folder_name = os.path.basename(os.path.normpath(log_folder))
            filename = f'{folder_name}_training_progress.png'
        else:
            filename = 'aggregated_training_progress.png'
        
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    plt.show()


def analyze_training_run(log_folder, output_dir):
    aggregated_metrics = aggregate_metrics(log_folder)
        
    # Plot the aggregated metrics - pass log_folder to the function
    log_folder = os.path.basename(log_folder)
    plot_aggregated_metrics(aggregated_metrics, output_dir, log_folder)
    
    # Print summary statistics
    total_runs = len(glob.glob(os.path.join(log_folder, '**/training.txt'), recursive=True))
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

def compare_training_scenarios(scenario_metrics, output_dir=None):
    """
    Compare metrics across different training scenarios.
    
    Args:
        scenario_metrics: Dictionary mapping scenario names to their aggregated metrics
        output_dir: Directory to save the comparison plot
    """
    plt.figure(figsize=(15, 6))
    
    # Plot total reward
    plt.subplot(1, 2, 1)
    for scenario_name, metrics in scenario_metrics.items():
        iterations = metrics['iterations']
        mean_values = np.array(metrics['metrics']['total_rewards']['mean'])
        min_values = np.array(metrics['metrics']['total_rewards']['min'])
        max_values = np.array(metrics['metrics']['total_rewards']['max'])
        
        # Filter out None values
        valid_indices = ~np.isnan(mean_values)
        valid_iterations = np.array(iterations)[valid_indices]
        valid_mean = mean_values[valid_indices]
        valid_min = min_values[valid_indices]
        valid_max = max_values[valid_indices]
        
        plt.plot(valid_iterations, valid_mean, linewidth=2, label=scenario_name)
        plt.fill_between(valid_iterations, valid_min, valid_max, alpha=0.2)
    
    plt.title('Mean Total Reward Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot episode length
    plt.subplot(1, 2, 2)
    for scenario_name, metrics in scenario_metrics.items():
        iterations = metrics['iterations']
        mean_values = np.array(metrics['metrics']['episode_lengths']['mean'])
        min_values = np.array(metrics['metrics']['episode_lengths']['min'])
        max_values = np.array(metrics['metrics']['episode_lengths']['max'])
        
        # Filter out None values
        valid_indices = ~np.isnan(mean_values)
        valid_iterations = np.array(iterations)[valid_indices]
        valid_mean = mean_values[valid_indices]
        valid_min = min_values[valid_indices]
        valid_max = max_values[valid_indices]
        
        plt.plot(valid_iterations, valid_mean, linewidth=2, label=scenario_name)
        plt.fill_between(valid_iterations, valid_min, valid_max, alpha=0.2)
    
    plt.title('Mean Episode Length Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = 'scenario_comparison.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        print(f"Comparison plot saved to {os.path.join(output_dir, filename)}")
    
    plt.show()

def aggregate_eval_metrics(eval_folder):
    """
    Aggregate evaluation metrics from eval_results.txt files in a folder.
    
    Args:
        eval_folder: Path to the folder containing evaluation results
        
    Returns:
        Dictionary containing aggregated evaluation metrics
    """
    # Find all eval_results.txt files in subdirectories
    eval_files = glob.glob(os.path.join(eval_folder, '**/eval_results.txt'), recursive=True)
    
    if not eval_files:
        raise ValueError(f"No eval_results.txt files found in {eval_folder}")
    
    print(f"Found {len(eval_files)} evaluation result files to analyze")
    
    # Initialize aggregated metrics
    aggregated = {
        'rewards': [],
        'lengths': []
    }
    
    # Process each evaluation file
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                # Skip header line
                header = f.readline().strip()
                columns = header.split(',')
                
                # Check if file has command informatio
                
                # Process each line
                for line in f:
                    values = line.strip().split(',')
                    if len(values) >= 3:  # At minimum we need rollout, reward, length
                        reward = float(values[1])
                        length = int(values[2])
                        
                        aggregated['rewards'].append(reward)
                        aggregated['lengths'].append(length)
                        
            
            print(f"Processed {eval_file}")
        except Exception as e:
            print(f"Error processing {eval_file}: {e}")
    
    # Calculate statistics
    if aggregated['rewards']:
        aggregated['mean_reward'] = np.mean(aggregated['rewards'])
        aggregated['std_reward'] = np.std(aggregated['rewards'])
        aggregated['min_reward'] = np.min(aggregated['rewards'])
        aggregated['max_reward'] = np.max(aggregated['rewards'])
        aggregated['median_reward'] = np.median(aggregated['rewards'])
        
        aggregated['mean_length'] = np.mean(aggregated['lengths'])
        aggregated['std_length'] = np.std(aggregated['lengths'])
        aggregated['min_length'] = np.min(aggregated['lengths'])
        aggregated['max_length'] = np.max(aggregated['lengths'])
        aggregated['median_length'] = np.median(aggregated['lengths'])
        
        # Every field in aggregated printed in a nicely formatted table
        print(f"\nAggregated {len(aggregated['rewards'])} evaluation rollouts")
        print(f"Mean reward: {aggregated['mean_reward']:.4f} ± {aggregated['std_reward']:.4f}")
        print(f"Min reward: {aggregated['min_reward']:.4f}")
        print(f"Max reward: {aggregated['max_reward']:.4f}")
        print(f"Median reward: {aggregated['median_reward']:.4f}")

        print(f"Mean episode length: {aggregated['mean_length']:.1f} ± {aggregated['std_length']:.1f}")
        print(f"Min episode length: {aggregated['min_length']}")
        print(f"Max episode length: {aggregated['max_length']}")
        print(f"Median episode length: {aggregated['median_length']:.1f}")
        print(f"Min reward: {aggregated['min_reward']:.4f}\n")

    else:
        print("No valid evaluation data found")
    
    return aggregated

def plot_eval_metrics(eval_metrics, output_dir=None, scenario_name=None):
    """
    Plot evaluation metrics with improved visualizations.
    
    Args:
        eval_metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save the plot
        scenario_name: Name of the scenario for the plot title
    """
    if not eval_metrics['rewards']:
        print("No evaluation data to plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. Scatter plot of reward vs episode length
    plt.subplot(2, 2, 1)
    plt.scatter(eval_metrics['lengths'], eval_metrics['rewards'], alpha=0.7, c='blue', edgecolors='k')
    plt.axhline(eval_metrics['mean_reward'], color='r', linestyle='--', label=f"Mean reward: {eval_metrics['mean_reward']:.2f}")
    plt.axvline(eval_metrics['mean_length'], color='g', linestyle='--', label=f"Mean length: {eval_metrics['mean_length']:.1f}")
    plt.title('Reward vs Episode Length')
    plt.xlabel('Episode Length (steps)')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Reward distribution with kernel density estimate
    plt.subplot(2, 2, 2)
    sns.histplot(eval_metrics['rewards'], kde=True, bins=20, color='blue', alpha=0.6)
    plt.axvline(eval_metrics['mean_reward'], color='r', linestyle='--', linewidth=2, 
                label=f"Mean: {eval_metrics['mean_reward']:.2f}")
    plt.axvline(eval_metrics['median_reward'], color='g', linestyle='--', linewidth=2, 
                label=f"Median: {eval_metrics['median_reward']:.2f}")
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Episode length distribution
    plt.subplot(2, 2, 3)
    sns.histplot(eval_metrics['lengths'], kde=True, bins=20, color='green', alpha=0.6)
    plt.axvline(eval_metrics['mean_length'], color='r', linestyle='--', linewidth=2, 
                label=f"Mean: {eval_metrics['mean_length']:.1f}")
    plt.axvline(eval_metrics['median_length'], color='b', linestyle='--', linewidth=2, 
                label=f"Median: {eval_metrics['median_length']:.1f}")
    plt.title('Episode Length Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Performance summary as a table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = (
        f"EVALUATION SUMMARY\n\n"
        f"Total Episodes: {len(eval_metrics['rewards'])}\n\n"
        f"REWARD STATISTICS:\n"
        f"  Mean:   {eval_metrics['mean_reward']:.4f}\n"
        f"  Median: {eval_metrics['median_reward']:.4f}\n"
        f"  Std:    {eval_metrics['std_reward']:.4f}\n"
        f"  Min:    {eval_metrics['min_reward']:.4f}\n"
        f"  Max:    {eval_metrics['max_reward']:.4f}\n\n"
        f"EPISODE LENGTH STATISTICS:\n"
        f"  Mean:   {eval_metrics['mean_length']:.1f}\n"
        f"  Median: {eval_metrics['median_length']:.1f}\n"
        f"  Std:    {eval_metrics['std_length']:.1f}\n"
        f"  Min:    {eval_metrics['min_length']}\n"
        f"  Max:    {eval_metrics['max_length']}"
    )
    print(summary_text)
    plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', 
             verticalalignment='center', horizontalalignment='left')
    
    plt.tight_layout()
    
    # Add overall title if scenario name is provided
    if scenario_name:
        plt.suptitle(f"Evaluation Metrics for {scenario_name}", fontsize=16)
        plt.subplots_adjust(top=0.92)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{scenario_name}_eval_metrics.png" if scenario_name else "eval_metrics.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        print(f"Plot saved to {os.path.join(output_dir, filename)}")
    
    plt.show()

def compare_eval_metrics(scenario_metrics, output_dir=None):
    """
    Compare evaluation metrics across different scenarios with improved visualizations.
    
    Args:
        scenario_metrics: Dictionary mapping scenario names to their evaluation metrics
        output_dir: Directory to save the comparison plot
    """
    if not scenario_metrics:
        print("No scenarios to compare")
        return
    
    # Prepare data
    scenarios = []
    mean_rewards = []
    std_rewards = []
    median_rewards = []
    mean_lengths = []
    std_lengths = []
    
    reward_data = []
    length_data = []
    
    for scenario_name, metrics in scenario_metrics.items():
        if metrics['rewards']:
            scenarios.append(scenario_name)
            mean_rewards.append(metrics['mean_reward'])
            std_rewards.append(metrics['std_reward'])
            median_rewards.append(metrics['median_reward'])
            mean_lengths.append(metrics['mean_length'])
            std_lengths.append(metrics['std_length'])
            reward_data.append(metrics['rewards'])
            length_data.append(metrics['lengths'])
    
    if not scenarios:
        print("No valid data to compare")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. Bar chart of mean rewards with error bars
    plt.subplot(2, 2, 1)
    x = np.arange(len(scenarios))
    plt.bar(x, mean_rewards, yerr=std_rewards, alpha=0.7, capsize=10, color='skyblue', edgecolor='black')
    plt.xticks(x, scenarios, rotation=45, ha='right')
    plt.title('Mean Reward by Scenario')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 2. Violin plots for reward distributions
    plt.subplot(2, 2, 2)
    violin_parts = plt.violinplot(reward_data, showmeans=True, showmedians=True)
    # Customize violin plot
    for pc in violin_parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    plt.xticks(np.arange(1, len(scenarios) + 1), scenarios, rotation=45, ha='right')
    plt.title('Reward Distribution by Scenario')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. Bar chart of mean episode lengths
    plt.subplot(2, 2, 3)
    plt.bar(x, mean_lengths, yerr=std_lengths, alpha=0.7, capsize=10, color='lightgreen', edgecolor='black')
    plt.xticks(x, scenarios, rotation=45, ha='right')
    plt.title('Mean Episode Length by Scenario')
    plt.ylabel('Mean Episode Length')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Scenario', 'Mean Reward', 'Median Reward', 'Mean Length']
    for i, scenario in enumerate(scenarios):
        table_data.append([
            scenario, 
            f"{mean_rewards[i]:.2f} ± {std_rewards[i]:.2f}", 
            f"{median_rewards[i]:.2f}",
            f"{mean_lengths[i]:.1f} ± {std_lengths[i]:.1f}"
        ])
    
    # Create the table
    table = plt.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.suptitle('Scenario Comparison', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = "eval_metrics_comparison.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        print(f"Comparison plot saved to {os.path.join(output_dir, filename)}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize ZBot training progress')
    parser.add_argument('--log_folder', type=str, default='results/zbot-walking',
                        help='Path to the folder containing training runs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--scenarios', nargs='+', type=str, default=None,
                        help='List of scenario names to compare')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Process evaluation results instead of training logs')
    args = parser.parse_args()
    
    if args.scenarios:
        # Compare multiple scenarios
        scenario_metrics = {}
        for scenario in args.scenarios:
            scenario_log_folder = os.path.join(args.log_folder, scenario)
            print(f"Processing scenario: {scenario}")
            try:
                if args.eval:
                    scenario_metrics[scenario] = aggregate_eval_metrics(scenario_log_folder)
                else:
                    scenario_metrics[scenario] = aggregate_metrics(scenario_log_folder)
            except Exception as e:
                print(f"Error processing scenario {scenario}: {e}")
        
        if args.eval:
            if scenario_metrics:
                compare_eval_metrics(scenario_metrics, args.output_dir)
            else:
                print("No valid scenarios to compare")
        else:
            if scenario_metrics:
                compare_training_scenarios(scenario_metrics, args.output_dir)
            else:
                print("No valid scenarios to compare")
    else:
        # Analyze a single training run or evaluation
        if args.eval:
            try:
                eval_metrics = aggregate_eval_metrics(args.log_folder)
                plot_eval_metrics(eval_metrics, args.output_dir)
            except Exception as e:
                print(f"Error processing evaluation results: {e}")
        else:
            analyze_training_run(args.log_folder, args.output_dir)
        
if __name__ == "__main__":
    main() 