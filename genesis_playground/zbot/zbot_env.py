""" ZBot environment """
import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def cosine_interpolation(start, end, t, max_t):
    return start + (end - start) * (1 - math.cos(t * math.pi / max_t)) / 2

def linear_interpolation(start, end, t, max_t):
    return start + (end - start) * (t / max_t)

def get_from_curriculum(curriculum, t, max_t):
    min_start = curriculum["start"][0]
    min_end = curriculum["end"][0]
    max_start = curriculum["start"][1]
    max_end = curriculum["end"][1]
    min_value = linear_interpolation(min_start, min_end, t, max_t)
    max_value = linear_interpolation(max_start, max_end, t, max_t)
    return np.random.uniform(min_value, max_value)

class ZbotEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="mps"):
        self.device = torch.device(device)
        self.total_steps = 0
        self.max_steps = 40_000_000

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2), # substep=2 for 50hz
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=80,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="genesis_playground/resources/zbot/robot_fixed.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        kp_values = [self.env_cfg["kp"]] * self.num_actions
        kv_values = [self.env_cfg["kd"]] * self.num_actions
        self.robot.set_dofs_kp(kp_values, self.motor_dofs)
        self.robot.set_dofs_kv(kv_values, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        # initialize buffers
        
        # feet air time
        self.foot_link_indices = [
            10, 11
        ]

        # Create buffers to track how long each foot has been in the air.
        # Shape: [num_envs, num_feet].
        self.feet_air_time = torch.zeros((self.num_envs, len(self.foot_link_indices)),
                                         device=self.device, dtype=torch.float)
        
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        
        torques = self.robot.get_dofs_control_force(self.motor_dofs)
        max_torque = self.env_cfg["max_torque"]
        # RFI https://arxiv.org/pdf/2209.12878
        noise = (2.0 * torch.rand_like(torques) - 1.0) * self.env_cfg["rfi_scale"]
        torques = torques + noise
        torques = torch.clamp(torques, -max_torque, max_torque)
        self.robot.control_dofs_force(torques, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        
        contacts = self.robot.get_contacts()
        # foot_contact[e, f] = True if foot f in env e is on the floor.
        foot_contact = torch.zeros(
            (self.num_envs, len(self.foot_link_indices)),
            device=self.device,
            dtype=torch.bool
        )
        
        plane_link_index = 0
        
        link_a = contacts["link_a"]  # shape [num_envs, num_contacts]
        link_b = contacts["link_b"]  # shape [num_envs, num_contacts]
        
        # Create a mask for each foot link index
        for f_idx, foot_li in enumerate(self.foot_link_indices):
            # Check if either link_a or link_b matches foot_li and the other matches plane_link_index
            foot_plane_contact = ((link_a == foot_li) & (link_b == plane_link_index)) | ((link_b == foot_li) & (link_a == plane_link_index))
            # Any contact along the contact dimension means that foot is in contact
            foot_contact[:, f_idx] = torch.tensor(foot_plane_contact.any(1))
                        
        not_in_contact = ~foot_contact
        self.feet_air_time += (not_in_contact * self.dt)
        self.feet_air_time[foot_contact] = 0.0

        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.total_steps += self.num_envs
        
        # during training rsl_rl expects the critic observations to be 
        # returned in this format
        self.extras["observations"] = {"critic": self.obs_buf}

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        # The latest rsl_rl expects seperate observation vectors for our policy and 
        # our critic. For now, I'm just returning the same observation vector for both.

        # This means that when we try to train a minimal policy, we can exclude certain
        # observations from our policy, but still give them to the critic defining our 
        # value function target. This will be interesting to leverage
        return self.obs_buf, {"observations": {"critic": self.obs_buf}}

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # Sample uniform multipliers between min and max
        kp_mult = gs_rand_float(
            self.env_cfg["kp_multipliers"][0],
            self.env_cfg["kp_multipliers"][1],
            (1,),
            self.device
        )
        kd_mult = gs_rand_float(
            self.env_cfg["kd_multipliers"][0], 
            self.env_cfg["kd_multipliers"][1],
            (1,),
            self.device
        )

        # Apply multipliers to default values
        kp_values = torch.full(
            (self.num_actions,),
            self.env_cfg["kp"] * kp_mult.item(),
            device=self.device
        )
        kd_values = torch.full(
            (self.num_actions,),
            self.env_cfg["kd"] * kd_mult.item(), 
            device=self.device
        )

        # Set the PD gains
        self.robot.set_dofs_kp(kp_values, self.motor_dofs)
        self.robot.set_dofs_kv(kd_values, self.motor_dofs)
        
        # friction
        friction = get_from_curriculum(self.env_cfg["env_friction_range"], self.total_steps, self.max_steps)
        self.robot.set_friction(friction)
        self.plane.set_friction(friction)
        
        # link mass
        link_mass_mult = get_from_curriculum(self.env_cfg["link_mass_multipliers"], self.total_steps, self.max_steps)
        for link in self.robot.links:
            link.set_mass(link.get_mass() * link_mass_mult)

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        # Reset environments in chunks of 32 for better efficiency
        chunk_size = 32
        for i in range(0, self.num_envs, chunk_size):
            chunk_end = min(i + chunk_size, self.num_envs)
            chunk_indices = torch.arange(i, chunk_end, device=self.device)
            self.reset_idx(chunk_indices)
        return self.obs_buf, None

    # ------------ reward functions----------------

    # Maybe there is room to experiement with different reward functions? We can modify
    # terms of the reward function or add our own pretty easily here

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_gait_symmetry(self):
        # Reward symmetric gait patterns
        left_hip = self.dof_pos[:, self.env_cfg["dof_names"].index("L_Hip_Pitch")]
        right_hip = self.dof_pos[:, self.env_cfg["dof_names"].index("R_Hip_Pitch")]
        left_knee = self.dof_pos[:, self.env_cfg["dof_names"].index("L_Knee_Pitch")]
        right_knee = self.dof_pos[:, self.env_cfg["dof_names"].index("R_Knee_Pitch")]
        
        hip_symmetry = torch.abs(left_hip - right_hip)
        knee_symmetry = torch.abs(left_knee - right_knee)
        
        return torch.exp(-(hip_symmetry + knee_symmetry))

    def _reward_energy_efficiency(self):
        # Reward energy efficiency by penalizing high joint velocities
        return -torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_feet_air_time(self):
        """
        Mirrors logic from joystick.py: reward for how long each foot is in the air.
        We apply a threshold window [threshold_min, threshold_max], then
        the reward is gained on first contact.
        """
        threshold_min = 0.01
        threshold_max = 0.5

        cmd_norm = torch.linalg.norm(self.commands, dim=1)
        clipped = (self.feet_air_time - threshold_min).clamp(min=0.0, max=threshold_max - threshold_min)
        reward = clipped.sum(dim=1)
        zero_mask = (cmd_norm <= 0.1)
        reward[zero_mask] = 0.0
        return reward
