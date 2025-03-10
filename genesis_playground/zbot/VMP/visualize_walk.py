"""Visualize the walking of the robot."""
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# Load CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Plot joint movements
def plot_joint_movements(df):
    plt.figure(figsize=(12, 6))
    
    # Select joints to plot
    joints = [
        'left_hip_yaw',
        'left_hip_roll', 
        'left_hip_pitch', 
        'left_knee', 
        'right_hip_yaw', 
        'right_hip_roll', 
        'right_hip_pitch', 
        'right_knee', 
        'right_ankle',
        'left_ankle']
    df_filtered = df[df['joint_name'].isin(joints)]
    
    for joint in joints:
        joint_df = df_filtered[df_filtered['joint_name'] == joint]
        plt.plot(joint_df['elapsed_time'], joint_df['position'], label=joint)
    
    plt.xlabel('Time')
    plt.ylabel('Joint Position')
    plt.title('Robot Walking - Joint Velocities Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize robot walking data from CSV file.')
    
    parser.add_argument(
        '--file_path', '-f',
        type=str,
        default='data/walking_test_20250304_210109.csv',
        help='Path to the CSV file containing robot walking data'
    )
    args = parser.parse_args()
    
    
    df = load_data(args.file_path)
    plot_joint_movements(df)
