""" ZBot environment with VMP conditioning """
import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from VMP.VMP import WalkingVAE  # Your existing VAE implementation
from VMP.utils import load_walking_data  # Your existing data loader
from zbot_env import ZbotEnv
import os
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import glob


class ZbotEnvVMP(ZbotEnv):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, 
                 vae_model_path=None, show_viewer=False, device="cuda"):
        # Add motion tracking reward weights to reward_cfg
        reward_cfg.update({
            "pos_weight": 1.0,        # Weight for joint position tracking
            "vel_weight": 0.1,        # Weight for joint velocity tracking
            "root_pos_weight": 0.5,   # Weight for root position tracking
            "root_rot_weight": 0.5    # Weight for root rotation tracking
        })
        
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, 
                        show_viewer=show_viewer, device=device)
        
        # VMP-specific initialization
        self.window_size = env_cfg.get("window_size", 15)  # Must match VMP training config
        self.motion_window_size = 2 * self.window_size + 1  # Actual window size for VAE
        self.latent_dim = 64   # Must match VMP training config
        self.motion_dim = len(env_cfg["selected_joints"]) * 2  # pos + vel
        
        # Motion sampling attributes
        self.current_start_idx = 0
        self.current_clip = None
        self.current_window = None
        self.current_z = None
        self.current_m = None
        self.step_count = 0  # Initialize step counter
        
        # Store reward weights for motion tracking
        self.motion_reward_weights = {
            "pos_weight": reward_cfg.get("pos_weight", 1.0),
            "vel_weight": reward_cfg.get("vel_weight", 0.1),
            "root_pos_weight": reward_cfg.get("root_pos_weight", 0.5),
            "root_rot_weight": reward_cfg.get("root_rot_weight", 0.5)
        }
        
        # Load VAE model if provided
        if vae_model_path:
            self.vae = self._load_vae(vae_model_path)
            self.vae.eval()
            self._init_motion_reference()
            
            # Adjust observation space
            self.original_obs_dim = obs_cfg["num_obs"]  # Store original dimension
            obs_cfg["num_obs"] += self.latent_dim + self.motion_dim
            self._update_observation_space()  # This must recreate actor_obs_mapping
        
        # Domain randomization parameters from Disney paper
        self.domain_rand_cfg = {
            "mass_range": [0.8, 1.2],
            "friction_range": [0.5, 1.5],
            "joint_noise": 0.05,
            "push_interval": 2.0,
            "push_force_range": [5, 20]
        }

    def populate_observation_buffers(self):
        """Populate observation buffers with current state"""
        # First create base observations
        base_obs = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 10
                self.dof_vel * self.obs_scales["dof_vel"],  # 10
                self.actions,  # 10
            ],
            dim=-1
        )
        
        # Initialize full observation buffer with base observations
        self.full_obs_buf = base_obs
        
        # Append VMP data if available
        if hasattr(self, 'vae') and self.vae is not None and self.current_m is not None and self.current_z is not None:
            # Convert to tensors and ensure correct shape (num_envs, dim)
            # Both current_m and current_z are already tensors, just need to reshape and expand
            current_m_tensor = self.current_m.view(1, -1).expand(self.num_envs, -1)
            current_z_tensor = self.current_z.view(1, -1).expand(self.num_envs, -1)
            
            # Scale the VMP data
            current_m_tensor = current_m_tensor * self.obs_scales.get("motion", 1.0)
            current_z_tensor = current_z_tensor * self.obs_scales.get("latent", 1.0)
            
            # Concatenate VMP data
            self.full_obs_buf = torch.cat([
                self.full_obs_buf,
                current_m_tensor,
                current_z_tensor
            ], dim=-1)
        
        # Verify dimensions
        assert self.full_obs_buf.shape[1] == self.num_obs, \
            f"Observation buffer has {self.full_obs_buf.shape[1]} dims but expected {self.num_obs}"
        assert self.actor_obs_mapping.max() < self.full_obs_buf.shape[1], \
            f"Actor obs mapping max index {self.actor_obs_mapping.max()} exceeds buffer dims {self.full_obs_buf.shape[1]}"
        
        # Update actor and critic observation buffers
        self.actor_obs_buf = self.full_obs_buf[:, self.actor_obs_mapping]
        self.critic_obs_buf = self.full_obs_buf 

    def _load_vae(self, model_path):
        """Load pretrained VAE model with proper configuration"""
        # Calculate input dimension for VAE
        num_joints = len(self.env_cfg["selected_joints"])
        input_dim = num_joints * 2  # pos + vel for each joint
        
        #print(f"Initializing VAE with: input_dim={input_dim}, window_size={self.window_size}, latent_dim={self.latent_dim}")
        
        model = WalkingVAE(
            input_dim=input_dim,
            window_size=self.window_size,
            latent_dim=self.latent_dim
        )
        
        # Load and verify model state
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Verify model's expected input shape
        expected_flat_dim = input_dim * self.window_size
        #print(f"Expected flat dimension: {expected_flat_dim}")
        
        return model

    def _load_motion_dataset(self):
        """Load motion dataset from CSV files"""
        # Get all CSV files in the data directory
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        csv_files = glob.glob(os.path.join(data_dir, "walking_test_*.csv"))
        
        if not csv_files:
            print("Warning: No CSV files found in data directory. Using synthetic data.")
            return self._generate_synthetic_data()
        
        # Load and process each CSV file
        motion_clips = []
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Get unique timestamps to group data by frame
                timestamps = df['timestamp'].unique()
                num_frames = len(timestamps)
                num_joints = len(self.env_cfg["selected_joints"])
                
                # Initialize motion data tensor
                motion_data = torch.zeros((num_frames, num_joints * 2), device=self.device)
                
                # Process each frame
                for i, timestamp in enumerate(timestamps):
                    frame_data = df[df['timestamp'] == timestamp]
                    
                    # Extract joint positions and velocities
                    for j, joint_name in enumerate(self.env_cfg["selected_joints"]):
                        joint_data = frame_data[frame_data['joint_name'] == joint_name]
                        if not joint_data.empty:
                            # Convert to tensor and normalize
                            pos_data = torch.tensor(joint_data['position'].values[0], device=self.device)
                            vel_data = torch.tensor(joint_data['velocity'].values[0], device=self.device)
                            
                            # Store in motion_data
                            motion_data[i, j] = pos_data
                            motion_data[i, j + num_joints] = vel_data
                
                motion_clips.append(motion_data)
                print(f"Loaded {csv_file} with {num_frames} frames")
                
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
                continue
        
        if not motion_clips:
            print("Warning: No valid motion clips loaded. Using synthetic data.")
            return self._generate_synthetic_data()
        
        # Concatenate all clips
        motion_data = torch.cat(motion_clips, dim=0)
        
        # Normalize the data
        self.motion_scaler = RobustScaler()
        motion_data_flat = motion_data.cpu().numpy()
        motion_data_normalized = self.motion_scaler.fit_transform(motion_data_flat)
        motion_data = torch.from_numpy(motion_data_normalized).float().to(self.device)
        
        print(f"Loaded {len(motion_clips)} motion clips with {motion_data.shape[0]} total frames")
        return [motion_data]
    
    def _generate_synthetic_data(self):
        """Generate synthetic data when no real data is available"""
        num_frames = 1000
        num_joints = len(self.env_cfg["selected_joints"])
        motion_dim = num_joints * 2  # pos + vel
        
        # Generate structured synthetic data
        t = torch.linspace(0, 2*np.pi, num_frames, device=self.device)
        
        # Create periodic motion patterns
        synthetic_data = torch.zeros(num_frames, motion_dim, device=self.device)
        
        # Add periodic patterns for each joint
        for i in range(num_joints):
            # Position patterns
            synthetic_data[:, i] = torch.sin(t + i) * 0.5
            # Velocity patterns (derivative of position)
            synthetic_data[:, i + num_joints] = torch.cos(t + i) * 0.5
        
        # Add some random variation
        synthetic_data += torch.randn_like(synthetic_data) * 0.1
        
        # Normalize the data
        self.motion_scaler = RobustScaler()
        motion_data_flat = synthetic_data.cpu().numpy()
        motion_data_normalized = self.motion_scaler.fit_transform(motion_data_flat)
        synthetic_data = torch.from_numpy(motion_data_normalized).float().to(self.device)
        
        return [synthetic_data]

    def _init_motion_reference(self):
        """Initialize motion reference data"""
        # Load motion dataset (implement this based on your data loading logic)
        self.motion_dataset = self._load_motion_dataset()
        if self.motion_dataset:
            # Initialize with first clip
            self.current_clip = self.motion_dataset[0]
            self.current_start_idx = 0
            self._sample_next_frame()

    def _sample_next_frame(self):
        """Sample the next motion frame with improved temporal continuity"""
        if self.current_clip is None:
            return None
        
        # Calculate next start index with overlap
        overlap = self.window_size // 2  # 50% overlap between windows
        next_start_idx = (self.current_start_idx + overlap) % len(self.current_clip)
        
        # Get current window
        end_idx = (next_start_idx + self.motion_window_size) % len(self.current_clip)
        if end_idx > next_start_idx:
            new_window = self.current_clip[next_start_idx:end_idx]
        else:
            # Handle wrap-around with smooth transition
            first_part = self.current_clip[next_start_idx:]
            second_part = self.current_clip[:end_idx]
            
            # Create smooth transition
            transition_size = min(len(first_part), len(second_part))
            if transition_size > 0:
                alpha = torch.linspace(1, 0, transition_size, device=self.device).view(-1, 1)
                transition = alpha * first_part[-transition_size:] + (1 - alpha) * second_part[:transition_size]
                new_window = torch.cat([first_part[:-transition_size], transition, second_part[transition_size:]])
            else:
                new_window = torch.cat([first_part, second_part])
        
        # Update indices
        self.current_start_idx = next_start_idx
        
        # Ensure window has correct shape
        if new_window.shape[0] != self.motion_window_size:
            if new_window.shape[0] < self.motion_window_size:
                # Pad with extrapolation
                pad_size = self.motion_window_size - new_window.shape[0]
                last_diff = new_window[-1] - new_window[-2]
                # Create padding with same dimensions as new_window
                padding = torch.stack([new_window[-1] + last_diff * (i+1) for i in range(pad_size)])
                new_window = torch.cat([new_window, padding])
            else:
                new_window = new_window[:self.motion_window_size]
        
        # Get latent encoding
        with torch.no_grad():
            # Reshape for VAE
            window_for_vae = new_window.unsqueeze(0)
            window_flat = window_for_vae.reshape(1, -1)
            
            # Encode
            self.current_z = self.vae.encode(window_flat)[0].squeeze(0)
            
            # Visualize reconstruction periodically
            if self.step_count % 100 == 0:  # Reduced visualization frequency
                # Decode
                reconstructed_flat = self.vae.decode(self.current_z.unsqueeze(0))
                reconstructed = reconstructed_flat.reshape(1, self.motion_window_size, self.motion_dim)
                
                # Denormalize for visualization
                # Reshape to match the scaler's expected input shape
                window_flat_np = window_flat.cpu().numpy()
                reconstructed_flat_np = reconstructed_flat.cpu().numpy()
                
                # Ensure the data has the correct number of features
                num_features = self.motion_dim  # This should match the number of features used during fitting
                window_flat_np = window_flat_np.reshape(-1, num_features)
                reconstructed_flat_np = reconstructed_flat_np.reshape(-1, num_features)
                
                # Apply inverse transform
                original_window = self.motion_scaler.inverse_transform(window_flat_np)
                reconstructed_window = self.motion_scaler.inverse_transform(reconstructed_flat_np)
                
                # Reshape back to window format
                original_window = original_window.reshape(1, self.motion_window_size, self.motion_dim)
                reconstructed_window = reconstructed_window.reshape(1, self.motion_window_size, self.motion_dim)
                
                # Plot comparison
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                for j in range(3):
                    plt.subplot(3, 2, 2*j+1)
                    plt.plot(original_window[0, :, j], label='Original')
                    plt.plot(reconstructed_window[0, :, j], label='Reconstructed')
                    plt.title(f"Joint {j} Position")
                    plt.legend()
                    
                    plt.subplot(3, 2, 2*j+2)
                    vel_idx = j + len(self.env_cfg["selected_joints"])
                    plt.plot(original_window[0, :, vel_idx], label='Original')
                    plt.plot(reconstructed_window[0, :, vel_idx], label='Reconstructed')
                    plt.title(f"Joint {j} Velocity")
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"vae_reconstruction_{self.step_count}.png")
                plt.close()
        
        # Get current frame as motion reference
        self.current_m = new_window[self.window_size].clone()
        
        return new_window

    def reset(self):
        # Reset step counter
        self.step_count = 0
        
        # Call parent reset
        super().reset()
        
        # Sample new motion window and encode
        if self.vae:
            self.current_window = self._sample_next_frame()
            
            # Validate window shape
            assert self.current_window.ndim == 2, "Window must be 2D"
            assert self.current_window.shape[0] == self.motion_window_size, "Incorrect window length"
            
            self.current_m = self.current_window[self.window_size]  # Current frame
            
            self.populate_observation_buffers()
        
        return self.actor_obs_buf

    def step(self, actions):
        # Update motion reference before parent's step()
        if self.vae:
            # Ensure window is initialized and valid
            if self.current_window is None or self.current_window.ndim != 2:
                self.reset()  # Force reinitialization if invalid
            
            # Store the current window for later use
            prev_window = self.current_window.clone()
            
            # Sample next frame
            new_window = self._sample_next_frame()
            if new_window is not None:
                # Update the window with the new frame
                self.current_window = new_window
            
            # Periodically re-encode
            if self.step_count % self.motion_window_size == 0:
                with torch.no_grad():
                    window_tensor = self.current_window.to(self.device)
                    # Keep z as tensor instead of converting to numpy
                    self.current_z = self.vae.encode(window_tensor.unsqueeze(0))[0].to(self.device)
            
            self.current_m = self.current_window[self.window_size]
        
        # Call parent's step AFTER updating motion reference
        obs, rew, done, info = super().step(actions)
        
        # Add debug prints for episode termination conditions
        # if self.step_count % 100 == 0:  # Print every 100 steps
        #     print(f"\nEpisode Termination Debug:")
        #     print(f"Step count: {self.step_count}")
        #     print(f"Episode length: {self.episode_length_buf.max().item()} / {self.max_episode_length}")
        #     print(f"Max pitch: {torch.abs(self.base_euler[:, 1]).max().item()} / {self.env_cfg['termination_if_pitch_greater_than']}")
        #     print(f"Max roll: {torch.abs(self.base_euler[:, 0]).max().item()} / {self.env_cfg['termination_if_roll_greater_than']}")
        #     print(f"Any reset: {done.any().item()}")
        #     print(f"Number of resets: {done.sum().item()}")
        
        # Force some environments to reset randomly (1% chance each step)
        random_resets = torch.rand(self.num_envs, device=self.device) < 0.01
        done = done | random_resets
        
        # Increment step counter
        self.step_count += 1
        
        return obs, rew, done, info

    def _update_observation_space(self):
        """Expand observation space with VMP conditioning"""
        # Get base observation labels
        base_labels = [
            "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",  # 3
            "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",  # 3
            "command_lin_vel_x", "command_lin_vel_y", "command_ang_vel_z",  # 3
            *[f"dof_pos_{name}" for name in self.env_cfg["dof_names"]],  # 10
            *[f"dof_vel_{name}" for name in self.env_cfg["dof_names"]],  # 10
            *[f"action_{name}" for name in self.env_cfg["dof_names"]]  # 10
        ]
        
        # Add VMP labels
        vmp_labels = []
        if hasattr(self, 'vae') and self.vae is not None:
            vmp_labels.extend([f"m_{i}" for i in range(self.motion_dim)])
            vmp_labels.extend([f"z_{i}" for i in range(self.latent_dim)])
        
        # Combine all labels
        self.obs_labels = base_labels + vmp_labels
        
        # Create the observation mask
        actor_obs_mask = torch.ones(len(self.obs_labels), dtype=torch.bool, device=self.device)
        for idx, label in enumerate(self.obs_labels):
            if label in self.obs_exclusions:
                actor_obs_mask[idx] = False
        
        # Update the mapping
        self.actor_obs_mapping = torch.where(actor_obs_mask)[0]
        
        # Update observation dimensions
        self.num_obs = len(self.obs_labels)

    def _reward_motion_tracking(self):
        if not self.vae:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Get reference values
        ref_pos = self.current_m[:len(self.env_cfg["selected_joints"])]
        ref_vel = self.current_m[len(self.env_cfg["selected_joints"]):]
        
        # Normalize reference values to match scale of actual values
        ref_pos = ref_pos * 0.1  # Scale down reference positions
        ref_vel = ref_vel * 0.005  # Scale down reference velocities more aggressively
        
        # Calculate tracking errors with proper normalization
        pos_error = torch.mean((self.dof_pos - ref_pos.unsqueeze(0))**2, dim=1) * 0.1  # Scale down position error
        vel_error = torch.mean((self.dof_vel - ref_vel.unsqueeze(0))**2, dim=1) * 0.005  # Scale down velocity error more aggressively
        
        # Handle root position and rotation errors with normalization
        root_pos_error = torch.mean((self.base_pos[:, :2] - ref_pos[:2].unsqueeze(0))**2, dim=1) * 0.2  # Increase root pos error scale
        root_rot_error = torch.mean((self.base_euler[:, 2:3] - ref_pos[2:3].unsqueeze(0))**2, dim=1) * 0.001  # Keep rotation error scale
        
        # Use weights from config but adjust them for better balance
        weights = {
            "pos_weight": self.motion_reward_weights["pos_weight"] * 2.0,  # Double position weight
            "vel_weight": self.motion_reward_weights["vel_weight"],  # Keep velocity weight
            "root_pos_weight": self.motion_reward_weights["root_pos_weight"] * 0.5,  # Reduce root position weight
            "root_rot_weight": self.motion_reward_weights["root_rot_weight"] * 0.01  # Keep root rotation weight
        }
        
        # Combine rewards using adjusted weights
        reward = -(
            weights["pos_weight"] * pos_error +
            weights["vel_weight"] * vel_error +
            weights["root_pos_weight"] * root_pos_error +
            weights["root_rot_weight"] * root_rot_error
        )
        
        # Scale up the entire motion tracking reward
        reward = reward * 50.0  # Increase from 10.0 to 50.0
        
        return reward

    def _apply_domain_randomization(self):
        """Apply Disney paper's domain randomization"""
        # Mass randomization
        masses = self.robot.get_link_masses()
        rand_masses = masses * torch.FloatTensor(self.num_envs, 1).uniform_(*self.domain_rand_cfg["mass_range"])
        self.robot.set_link_masses(rand_masses)
        
        # Friction randomization
        friction = torch.FloatTensor(self.num_envs).uniform_(*self.domain_rand_cfg["friction_range"])
        self.plane.set_friction(friction)
        
        # Joint noise
        joint_pos = self.robot.get_dofs_position()
        joint_pos += torch.randn_like(joint_pos) * self.domain_rand_cfg["joint_noise"]
        self.robot.set_dofs_position(joint_pos)
        
        # Random pushes
        if self.step_count % int(self.domain_rand_cfg["push_interval"] / self.dt) == 0:
            push_force = torch.FloatTensor(self.num_envs, 3).uniform_(*self.domain_rand_cfg["push_force_range"])
            self.robot.apply_force(push_force)

    def compute_reward(self):
        """Compute all rewards including base rewards and motion tracking"""
        # Get base rewards
        print("You are in the compute_reward", self._reward_tracking_lin_vel(), self._reward_tracking_ang_vel )
        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(),
            "tracking_ang_vel": self._reward_tracking_ang_vel(),
            "lin_vel_z": self._reward_lin_vel_z(),
            "base_height": self._reward_base_height(),
            "action_rate": self._reward_action_rate(),
            "similar_to_default": self._reward_similar_to_default(),
            "feet_air_time": self._reward_feet_air_time(),
        }
        
        # Add motion tracking reward if VAE is available
        if hasattr(self, 'vae') and self.vae is not None:
            motion_tracking = self._reward_motion_tracking()
            rewards["motion_tracking"] = motion_tracking
        
        # Scale rewards and store both raw and scaled values
        scaled_rewards = {}
        reward_info = {}
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Define reward scales to balance components
        reward_scales = {
            "tracking_lin_vel": 0.2,    # Further reduce base locomotion rewards
            "tracking_ang_vel": 0.2,
            "lin_vel_z": 0.2,
            "base_height": 0.2,
            "action_rate": 0.05,        # Further reduce action rate penalty
            "similar_to_default": 0.05,  # Further reduce default pose penalty
            "feet_air_time": 0.2,
            "motion_tracking": 1.0       # Keep motion tracking at full scale
        }
        
        for name, reward in rewards.items():
            # Get scale from our defined scales or config, default to 1.0
            scale = reward_scales.get(name, self.reward_cfg.get("reward_scales", {}).get(name, 1.0))
            
            # Store raw and scaled values
            reward_info[f"raw_{name}"] = reward.mean().item()
            scaled_reward = reward * scale
            reward_info[name] = scaled_reward.mean().item()
            scaled_rewards[name] = scaled_reward
            
            # Add to total
            total_reward += scaled_reward
        
        # Store episode statistics
        reward_info["total_reward"] = total_reward.mean().item()
        reward_info["episode_length"] = self.episode_length_buf.float().mean().item()
        
        return total_reward, reward_info