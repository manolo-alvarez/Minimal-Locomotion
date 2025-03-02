""" ZBot evaluation with keyboard controls using pygame. """

import argparse
import os
import pickle
import numpy as np
import torch
import pygame 
from zbot_env import ZbotEnv
from rsl_rl.runners import OnPolicyRunner
from policy_analyzer import PolicyAnalyzer  # Add this import at the module level

import genesis as gs

# Global dictionary to hold user commands for x, y, yaw.
USER_CMD = {"x": 0.0, "y": 0.0, "yaw": 0.0}
INCREMENT = 0.6

# Global flag to track if pygame is initialized
PYGAME_INITIALIZED = False

def init_pygame_window():
    """Initialize a small pygame window so that we can capture keyboard events."""
    global PYGAME_INITIALIZED
    if not PYGAME_INITIALIZED:
        pygame.init()
        PYGAME_INITIALIZED = True
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Keyboard Control")
    return screen

def handle_pygame_events():
    """
    Checks the pygame event queue and updates USER_CMD accordingly.
    W/S: forward/back
    A/D: left/right
    Q/E: yaw left/right
    """
    global USER_CMD
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                USER_CMD["x"] += INCREMENT
            elif event.key == pygame.K_s:
                USER_CMD["x"] -= INCREMENT
            elif event.key == pygame.K_a:
                USER_CMD["y"] += INCREMENT
            elif event.key == pygame.K_d:
                USER_CMD["y"] -= INCREMENT
            elif event.key == pygame.K_q:
                USER_CMD["yaw"] += INCREMENT
            elif event.key == pygame.K_e:
                USER_CMD["yaw"] -= INCREMENT
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                USER_CMD["x"] -= INCREMENT
            elif event.key == pygame.K_s:
                USER_CMD["x"] += INCREMENT
            elif event.key == pygame.K_a:
                USER_CMD["y"] -= INCREMENT
            elif event.key == pygame.K_d:
                USER_CMD["y"] += INCREMENT
            elif event.key == pygame.K_q:
                USER_CMD["yaw"] -= INCREMENT
            elif event.key == pygame.K_e:
                USER_CMD["yaw"] += INCREMENT

def update_pygame_display():
    """Update the pygame display with current commands."""
    if PYGAME_INITIALIZED:
        screen = pygame.display.get_surface()
        if screen:
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 24)
            text = font.render(f"CMD: x={USER_CMD['x']:.2f}, y={USER_CMD['y']:.2f}, yaw={USER_CMD['yaw']:.2f}", 
                             True, (255, 255, 255))
            screen.blit(text, (10, 10))
            pygame.display.flip()

def keyboard_control_policy(obs: torch.Tensor) -> torch.Tensor:
    """
    Returns actions for the robot based on USER_CMD (x, y, yaw).
    This function should modify the command inputs in the observation
    and then return a policy-generated action.
    """
    global policy  # Use the global policy from main()
    
    # Apply the user commands to the observation
    x_cmd = USER_CMD["x"]
    y_cmd = USER_CMD["y"]
    yaw_cmd = USER_CMD["yaw"]

    obs[:, 6] = x_cmd
    obs[:, 7] = y_cmd
    obs[:, 8] = yaw_cmd

    # Use the policy to generate actions based on the modified observation
    with torch.no_grad():
        actions = policy(obs)
    
    return actions

def apply_fixed_cmd(obs: torch.Tensor, fixed_cmd: dict) -> torch.Tensor:
    """
    Apply a fixed command to the robot.
    """
    x_cmd = fixed_cmd["x"]
    y_cmd = fixed_cmd["y"]
    yaw_cmd = fixed_cmd["yaw"]

    obs[:, 6] = x_cmd
    obs[:, 7] = y_cmd
    obs[:, 8] = yaw_cmd

    return obs

def deg2rad(deg):
    return deg * np.pi / 180

def run_sim(env, policy_fn, obs, use_keyboard=False, base_policy=None, screen=None):
    """
    Runs the simulation loop for a fixed timespan.
    
    Args:
        env: The environment
        policy_fn: Function that processes observations and returns actions
        obs: Initial observation
        use_keyboard: Whether to use keyboard control
        base_policy: Base policy to use for keyboard control
        screen: Pygame screen (passed from main thread when use_keyboard=True)
    """
    timesteps = 0
    max_timesteps = 2000
    
    # Continue simulation until stopped
    try:
        while timesteps < max_timesteps:
            if use_keyboard:
                # Use the USER_CMD global variable that's updated from the main thread
                x_cmd = USER_CMD["x"]
                y_cmd = USER_CMD["y"]
                yaw_cmd = USER_CMD["yaw"]

                obs[:, 6] = x_cmd
                obs[:, 7] = y_cmd
                obs[:, 8] = yaw_cmd
                
                # Use base policy to generate actions
                with torch.no_grad():
                    actions = base_policy(obs)
            else:
                # Apply fixed command and use policy
                fixed_cmd = {"x": 1.0, "y": 0.0, "yaw": 0.0}
                modified_obs = obs.clone()
                modified_obs[:, 6] = fixed_cmd["x"]
                modified_obs[:, 7] = fixed_cmd["y"]
                modified_obs[:, 8] = fixed_cmd["yaw"]
                
                with torch.no_grad():
                    actions = policy_fn(modified_obs)

            # Step the environment
            obs, rews, dones, infos = env.step(actions)
            timesteps += 1

    except Exception as e:
        print(f"Simulation error: {e}")

def analyze_policy(env, runner, save_dir="feature_analysis", num_samples=1000, device="cpu", seed=0):
    """
    Analyze the importance of features in the policy.
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get policy from runner
    policy = runner.get_inference_policy(device=device)
    
    # Get observation and action labels
    obs_labels = env.get_observation_labels()

    try:
        action_labels = env.get_action_labels()
    except AttributeError:
        action_labels = [f"Action_{i}" for i in range(env.num_actions)]

    # Create analyzer
    analyzer = PolicyAnalyzer(
        policy=policy,
        obs_labels=obs_labels,
        action_labels=action_labels,
        device=device,
        save_dir=f"{args.log_dir}/{args.exp_name}/feature_analysis"
    )

    # Collect observations using varying commands
    print(f"Collecting samples using multiple commands...")
    observations = collect_observations_with_varying_commands(
        env, policy, num_samples=args.analysis_samples
    )

    # Use robust analysis instead of standard analysis
    analyzer.robust_sensitivity_analysis(
        observations, 
        num_base_points=5,  # Sample 5 different base points for more robust results
        perturbation_scale=0.01
    )

    # Run the rest of the analysis
    analyzer.run_full_analysis(observations)
    
    return analyzer

def collect_observations_with_varying_commands(env, policy, num_samples=1000, seed=None, device="cpu"):
    """Collect observations using varying commands for diverse state coverage."""
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Define a variety of commands to try
    commands = [
        {"x": 1.0, "y": 0.0, "yaw": 0.0},   # Forward
        {"x": 0.0, "y": 1.0, "yaw": 0.0},   # Lateral
        {"x": 0.0, "y": 0.0, "yaw": 1.0},   # Turn
        {"x": 0.7, "y": 0.7, "yaw": 0.0},   # Diagonal
        {"x": -0.7, "y": 0.0, "yaw": 0.0},  # Backward
    ]
    
    print(f"Collecting {num_samples} samples with {len(commands)} different commands...")
    observations = []
    
    # Reset environment
    obs, _ = env.get_observations()
    
    for i in range(num_samples):
        # Cycle through different commands
        cmd = commands[i % len(commands)]
        
        # Apply command to observation
        obs_copy = obs.clone()
        
        # Apply the selected command
        obs_copy[:, 6] = cmd["x"]
        obs_copy[:, 7] = cmd["y"]
        obs_copy[:, 8] = cmd["yaw"]
        
        # Get action
        with torch.no_grad():
            action = policy(obs_copy)
        
        # Step environment
        result = env.step(action)
        
        # Handle different return formats
        if len(result) == 4:
            obs, _, _, _ = result
        elif len(result) == 5:
            obs, _, _, _, _ = result
        else:
            obs = result[0]
        
        # Store observation
        observations.append(obs_copy[0].clone())
        
        # Progress
        if (i+1) % (num_samples // 10) == 0:
            print(f"Collected {i+1}/{num_samples} samples")
    
    # Stack observations and move to the specified device
    return torch.stack(observations).to(device)

def collect_gait_data(env, policy, num_samples=1000, seed=None, device="cpu"):
    """
    Collect observations and foot contacts for gait phase analysis.
    
    Args:
        env: Environment instance
        policy: Policy function
        num_samples: Number of samples to collect
        seed: Random seed for reproducibility
        device: Device to run on
    
    Returns:
        observations: Tensor of observations
        foot_contacts: Tensor of foot contact information (if available)
    """
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"Collecting {num_samples} samples for gait analysis...")
    observations = []
    foot_contacts = []
    has_foot_contacts = hasattr(env, "get_foot_contacts")
    
    # Reset environment
    obs, _ = env.get_observations()
    
    # Use consistent command for gait analysis
    fixed_cmd = {"x": 1.0, "y": 0.0, "yaw": 0.0}  # Forward motion for consistent gait
    
    for i in range(num_samples):
        # Apply command to observation
        obs_copy = obs.clone()
        
        # Apply command (assuming standard ZBot indices)
        obs_copy[:, 6] = fixed_cmd["x"]
        obs_copy[:, 7] = fixed_cmd["y"]
        obs_copy[:, 8] = fixed_cmd["yaw"]
        
        # Get action
        with torch.no_grad():
            action = policy(obs_copy)
        
        # Collect foot contact data if available
        if has_foot_contacts:
            contacts = env.get_foot_contacts()
            foot_contacts.append(contacts)
        
        # Step environment
        result = env.step(action)
        
        # Handle different return formats
        if len(result) == 4:
            obs, _, _, _ = result
        elif len(result) == 5:
            obs, _, _, _, _ = result
        else:
            obs = result[0]
        
        # Store observation
        observations.append(obs_copy[0].clone())
        
        # Progress
        if (i+1) % (num_samples // 10) == 0:
            print(f"Collected {i+1}/{num_samples} samples")
    
    # Stack observations
    observations = torch.stack(observations).to(device)
    
    # Stack foot contacts if available
    if foot_contacts:
        foot_contacts = torch.stack(foot_contacts).to(device)
        print(f"Collected foot contacts with shape: {foot_contacts.shape}")
        return observations, foot_contacts
    else:
        print("No foot contact information available from environment")
        return observations, None

def set_camera_view(viewer, position=(1.0, -1.0, 1.0), lookat=(0.0, 0.0, 0.3), fov=60):
    """Set custom camera position for better robot visibility."""
    if viewer is None:
        print("Warning: No viewer available.")
        return False
        
    try:
        # Set camera parameters directly on the viewer
        viewer.camera.position = position
        viewer.camera.lookat = lookat
        viewer.camera.fov = fov
        return True
    except AttributeError as e:
        print(f"Warning: Unable to adjust camera settings: {e}")
        return False

def add_camera_keybindings(viewer):
    """Add camera keyboard controls to the viewer."""
    if viewer is None:
        return
    
    try:
        # Define camera positions/views
        camera_presets = {
            "1": {"position": (1.0, -1.0, 1.0), "lookat": (0.0, 0.0, 0.3), "name": "Side view"},
            "2": {"position": (0.0, -2.0, 1.0), "lookat": (0.0, 0.0, 0.3), "name": "Back view"},
            "3": {"position": (2.0, 0.0, 0.8), "lookat": (0.0, 0.0, 0.3), "name": "Side view 2"},
            "4": {"position": (0.5, 0.5, 1.5), "lookat": (0.0, 0.0, 0.3), "name": "Diagonal view"},
            "5": {"position": (0.0, 0.0, 2.5), "lookat": (0.0, 0.0, 0.0), "name": "Top-down view"},
        }
        
        # Add key bindings for camera presets
        for key, preset in camera_presets.items():
            camera_pos = preset["position"]
            camera_lookat = preset["lookat"]
            name = preset["name"]
            
            viewer.add_key_binding(
                key=key,
                callback=lambda pos=camera_pos, lookat=camera_lookat: set_viewer_camera(viewer, pos, lookat),
                description=f"Camera: {name}"
            )
        
        # Add key binding for help text
        viewer.add_key_binding(
            key="h",
            callback=lambda: toggle_help_text(viewer),
            description="Toggle help text"
        )
            
        return True
    except Exception as e:
        print(f"Warning: Could not add camera keybindings: {e}")
        return False

def set_viewer_camera(viewer, position, lookat):
    """Set the camera position and lookat point for the viewer."""
    if viewer is None:
        return
    
    try:
        viewer.camera.position = position
        viewer.camera.lookat = lookat
    except AttributeError as e:
        print(f"Warning: Could not set viewer camera: {e}")

# Global variable to track help text visibility
SHOW_HELP_TEXT = True

def toggle_help_text(viewer):
    """Toggle the visibility of help text."""
    global SHOW_HELP_TEXT
    SHOW_HELP_TEXT = not SHOW_HELP_TEXT

def camera_controls_callback(viewer):
    """Display camera controls and other helpful information."""
    global SHOW_HELP_TEXT
    
    if not SHOW_HELP_TEXT:
        # Only show minimal help if help text is toggled off
        viewer.add_text(
            "Press 'H' for help", 
            x=10, y=10, font_size=14, color=(1.0, 1.0, 1.0, 0.8)
        )
        return
    
    # Calculate current FPS
    fps = viewer.fps_counter.get_fps() if hasattr(viewer, "fps_counter") else 0
    
    # Display all help text
    lines = [
        "Camera Controls:",
        "- Middle Mouse: Rotate camera",
        "- Right Mouse: Pan camera",
        "- Scroll Wheel: Zoom in/out",
        "",
        "Camera Presets:",
        "- 1: Side view",
        "- 2: Back view",
        "- 3: Side view 2",
        "- 4: Diagonal view",
        "- 5: Top-down view",
        "",
        "Other Controls:",
        "- H: Toggle this help text",
        f"FPS: {fps:.1f}"
    ]
    
    y_pos = 10
    for line in lines:
        viewer.add_text(
            line, 
            x=10, y=y_pos, font_size=14, color=(1.0, 1.0, 1.0, 0.8)
        )
        y_pos += 20

def detect_gait_phases(observations, env, num_phases=8):
    """
    Detect gait phases from observations using time-based estimation.
    
    Args:
        observations: Tensor of observations
        env: Environment instance (not used in time-based estimation)
        num_phases: Number of phases to identify
        
    Returns:
        phase_indices: Dictionary mapping phase number to observation indices
    """
    # Estimate typical stride duration (in number of steps)
    # For quadrupeds at normal speed, ~20-30 steps per stride is common
    estimated_stride_steps = 20
    
    # Create phases based on time/step count
    phases = np.linspace(0, 2*np.pi, estimated_stride_steps, endpoint=False)
    phases = np.tile(phases, len(observations)//estimated_stride_steps + 1)[:len(observations)]
    
    # Convert to discrete phase buckets
    discrete_phases = (phases/(2*np.pi) * num_phases).astype(int) % num_phases
    
    # Group observation indices by phase
    phase_indices = {}
    for phase in range(num_phases):
        phase_indices[phase] = np.where(discrete_phases == phase)[0]
    
    return phase_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zbot-walking")
    parser.add_argument("--ckpt", type=int, default=299)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use_keyboard", action="store_true", help="Use keyboard for control")
    parser.add_argument("--analyze", action="store_true", help="Perform feature importance analysis")
    parser.add_argument("--analysis_samples", type=int, default=1000, 
                      help="Number of samples to collect for analysis")
    parser.add_argument("--analysis_type", type=str, default="standard", 
                      choices=["standard", "robust", "command_variation", "phase_aware"],
                      help="Type of feature analysis to perform")
    parser.add_argument("--analysis_seed", type=int, default=None,
                      help="Random seed for feature analysis (use None for random results)")
    parser.add_argument("--show_viewer", action="store_true", default=True,
                      help="Show the Genesis viewer")
    parser.add_argument("--log_dir", type=str, default="logs",
                      help="Directory to save logs")
    args = parser.parse_args()

    gs.init()

    log_dir = f"{args.log_dir}/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{args.log_dir}/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # Create environment with show_viewer=True directly
    env = ZbotEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,  # Set viewer directly
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    
    # Make policy accessible globally for keyboard_control_policy
    global policy
    policy = runner.get_inference_policy(device=args.device)
    
    # Reset environment and get initial observation
    obs, extras = env.get_observations()
    
    # If analyze flag is set, perform feature importance analysis
    if args.analyze:
        save_dir = f"{args.log_dir}/{args.exp_name}/feature_analysis"
        if args.analysis_seed is not None:
            save_dir += f"_seed{args.analysis_seed}"
        
        if args.analysis_type == "standard":
            print("Running standard feature analysis...")
            analyzer = analyze_policy(
                env, 
                runner, 
                save_dir=save_dir,
                num_samples=args.analysis_samples,
                device=args.device,
                seed=args.analysis_seed
            )
        
        elif args.analysis_type == "robust":
            print("Running robust feature analysis across multiple base points...")
            policy = runner.get_inference_policy(device=args.device)
            analyzer = PolicyAnalyzer(
                policy=policy,
                obs_labels=env.get_observation_labels(),
                action_labels=env.get_action_labels() if hasattr(env, "get_action_labels") else None,
                device=args.device,
                save_dir=save_dir
            )
            
            observations = collect_observations_with_varying_commands(
                env, 
                policy, 
                num_samples=args.analysis_samples, 
                seed=args.analysis_seed,
                device=args.device  # Pass the device explicitly
            )
            
            analyzer.robust_sensitivity_analysis(
                observations, 
                num_base_points=5, 
                perturbation_scale=0.01
            )
            
            analyzer.run_full_analysis(observations)
        
        elif args.analysis_type == "command_variation":
            print("Running feature analysis with systematic command variation...")
            # Implementation for command variation analysis
            # ...
        
        elif args.analysis_type == "phase_aware":
            print("Running phase-aware feature analysis across gait cycles...")
            policy = runner.get_inference_policy(device=args.device)
            analyzer = PolicyAnalyzer(
                policy=policy,
                obs_labels=env.get_observation_labels(),
                action_labels=env.get_action_labels() if hasattr(env, "get_action_labels") else None,
                device=args.device,
                save_dir=save_dir
            )
            
            # Collect observations and foot contacts
            print("Collecting gait data for phase-aware analysis...")
            observations, foot_contacts = collect_gait_data(
                env, 
                policy, 
                num_samples=args.analysis_samples * 2,  # Collect more samples for gait analysis
                seed=args.analysis_seed,
                device=args.device
            )
            
            # Run phase-aware analysis
            if foot_contacts is not None:
                print("Using foot contacts for accurate gait phase detection")
                # Detect phases using foot contact information
                phase_indices = analyzer.detect_gait_phases_from_contacts(observations, foot_contacts)
            else:
                # Fall back to time-based phase detection
                print("Falling back to time-based gait phase detection")
                phase_indices = detect_gait_phases(observations, env)
            
            # Run the analysis with the detected phases
            analyzer.phase_aware_sensitivity_analysis(
                observations, 
                env, 
                phase_indices=phase_indices,
                foot_contacts=foot_contacts  # Pass the foot contacts
            )
            
            # Also run standard analysis for comparison
            analyzer.run_full_analysis(observations)
        
        print(f"Feature analysis complete. Results saved to {save_dir}/")
        
        # Continue with normal evaluation or exit
        if not args.use_keyboard and not args.show_viewer:
            print("Analysis complete. Exiting.")
            return

    # Setup viewer when needed
    if args.show_viewer and hasattr(env.scene, "viewer") and env.scene.viewer is not None:
        # Set the camera controls callback
        env.scene.viewer.custom_render_callback = camera_controls_callback

    # Run simulation with selected policy
    if args.show_viewer:
        # Run simulation in a separate thread
        with torch.no_grad():
            gs.tools.run_in_another_thread(
                run_sim, 
                args=(env, policy, obs)
            )
            
            # Start the viewer in the main thread if available
            if hasattr(env.scene, "viewer") and env.scene.viewer is not None:
                print("Starting viewer with enhanced camera controls.")
                print("Press 'H' to toggle help text.")
                
                # Start the viewer
                env.scene.viewer.start()
            else:
                print("Warning: Scene viewer not available.")
    else:
        # Run without the viewer
        run_sim(env, policy, obs)
        
if __name__ == "__main__":
    main()