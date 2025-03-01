""" ZBot evaluation with keyboard controls using pygame. """

import argparse
import os
import pickle
import numpy as np
import torch
import pygame 
from zbot_env import ZbotEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# Global dictionary to hold user commands for x, y, yaw.
USER_CMD = {"x": 0.0, "y": 0.0, "yaw": 0.0}
INCREMENT = 0.6

def init_pygame_window():
    """Initialize a small pygame window so that we can capture keyboard events."""
    pygame.init()
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
                # Don't handle pygame events here - they're handled in the main thread
                # Just use the USER_CMD global variable that's updated from the main thread
                # Apply keyboard commands to observation
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

def analyze_policy(env, runner, save_dir="feature_analysis", num_samples=1000, device="cpu"):
    """
    Analyze the importance of features in the policy.
    
    Args:
        env: The environment
        runner: The policy runner
        save_dir: Directory to save analysis results
        num_samples: Number of samples to collect
        device: Device to run on
    """
    # Get policy from runner
    policy = runner.get_inference_policy(device=device)
    
    # Get observation and action labels
    obs_labels = env.get_observation_labels()
    
    # Action labels - try to get them from the environment or create generic ones
    try:
        action_labels = env.get_action_labels()
    except AttributeError:
        # Create generic action labels if not available from the environment
        action_labels = [f"Action_{i}" for i in range(env.num_actions)]
    
    # Create analyzer
    analyzer = PolicyAnalyzer(
        policy=policy,
        obs_labels=obs_labels,
        action_labels=action_labels,
        device=device,
        save_dir=save_dir
    )
    
    # Collect samples
    print(f"Collecting {num_samples} samples for analysis...")
    observations = []
    
    # Reset environment
    obs, _ = env.get_observations()
    
    for i in range(num_samples):
        # Get action
        with torch.no_grad():
            action = policy(obs)
        
        # Step environment
        obs, _, _, _, _ = env.step(action)
        
        # Store observation
        observations.append(obs[0].clone())  # Store first environment's observation
        
        # Progress
        if (i+1) % (num_samples // 10) == 0:
            print(f"Collected {i+1}/{num_samples} samples")
    
    # Stack observations
    observations = torch.stack(observations).to(device)
    
    # Run analysis
    analyzer.run_full_analysis(observations)
    
    return analyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="zbot-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use_keyboard", action="store_true", help="Use keyboard for control")
    parser.add_argument("--analyze", action="store_true", help="Perform feature importance analysis")
    parser.add_argument("--analysis_samples", type=int, default=1000, 
                      help="Number of samples to collect for analysis")
    parser.add_argument("--show_viewer", action="store_true", default=True,
                      help="Show the Genesis viewer")
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = ZbotEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=args.device)
    
    # Reset environment and get initial observation
    obs, extras = env.get_observations()
    
    # If analyze flag is set, perform feature importance analysis
    if args.analyze:
        # Import the policy analyzer
        from policy_analyzer import PolicyAnalyzer, analyze_policy
        
        # Run analysis
        analysis_dir = f"logs/{args.exp_name}/feature_analysis"
        analyzer = analyze_policy(
            env, 
            runner, 
            save_dir=analysis_dir,
            num_samples=args.analysis_samples,
            device=args.device
        )
        print(f"Feature analysis complete. Results saved to {analysis_dir}/")
        
        # Continue with normal evaluation or exit
        if not args.use_keyboard:
            print("Analysis complete. Exiting.")
            return

    # Run simulation with selected policy using the Genesis viewer
    if args.show_viewer:
        # Initialize pygame in the main thread if using keyboard control
        screen = None
        if args.use_keyboard:
            screen = init_pygame_window()
        
        # Run simulation in a separate thread
        with torch.no_grad():
            if args.use_keyboard:
                # Pass in keyboard control arguments but DON'T initialize pygame in the thread
                gs.tools.run_in_another_thread(
                    run_sim, 
                    args=(env, policy, obs, True, policy, None)
                )
            else:
                gs.tools.run_in_another_thread(
                    run_sim, 
                    args=(env, policy, obs)
                )
            
            # Start the viewer in the main thread
            if hasattr(env.scene, "viewer") and env.scene.viewer is not None:
                # If using keyboard control, we need to handle pygame events in the main thread
                if args.use_keyboard and screen is not None:
                    def custom_render_callback():
                        # Handle pygame events
                        handle_pygame_events()
                        
                        # Update pygame display
                        screen.fill((0, 0, 0))
                        font = pygame.font.Font(None, 24)
                        text = font.render(f"CMD: x={USER_CMD['x']:.2f}, y={USER_CMD['y']:.2f}, yaw={USER_CMD['yaw']:.2f}", 
                                         True, (255, 255, 255))
                        screen.blit(text, (10, 10))
                        pygame.display.flip()
                    
                    # Set custom rendering callback to handle pygame events
                    env.scene.viewer.custom_render_callback = custom_render_callback
                
                # Start the viewer
                env.scene.viewer.start()
            else:
                print("Warning: Scene viewer not available. Make sure show_viewer=True in environment creation.")
    else:
        # Run without the viewer (useful for headless systems or pure analysis)
        if args.use_keyboard:
            # For non-viewer mode, initialize pygame in the main thread
            screen = init_pygame_window()
            run_sim(env, policy, obs, use_keyboard=True, base_policy=policy, screen=screen)
        else:
            run_sim(env, policy, obs)
        
if __name__ == "__main__":
    main()