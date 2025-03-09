import torch
import torch.onnx
import argparse
import os
import re
from pathlib import Path
from rsl_rl.modules import ActorCritic

NUM_OBSERVATION_FEATURES = 39

def convert_pt_to_onnx(pt_path, onnx_path, obs_dim=NUM_OBSERVATION_FEATURES):
    """
    Convert a PyTorch model to ONNX format with dynamic batch dimension
    
    Args:
        pt_path: Path to the PyTorch model checkpoint
        onnx_path: Path where the ONNX model will be saved
        obs_dim: Observation dimension length (features)
    """
    # Load the model
    print(f"Loading PyTorch model from {pt_path}")

    # Load the saved checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu")
    
    # Print checkpoint keys to debug
    print("Checkpoint keys:", checkpoint.keys())
    
    # Create a new model instance with the same architecture
    # The standard architecture from rsl_rl is ActorCritic
    # We need to determine the action dimension
    if "model_state_dict" in checkpoint:
        # Find output layer to determine action dimension
        for key in checkpoint["model_state_dict"].keys():
            if "actor" in key and "weight" in key:
                output_layer_key = key
        
        output_layer_shape = checkpoint["model_state_dict"][output_layer_key].shape
        num_actions = output_layer_shape[0]  # Output dimension of the last layer
        
        # Create model with same architecture as used in training
        model = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=[512, 256, 128],  # Default from zbot_train.py
            critic_hidden_dims=[512, 256, 128],  # Default from zbot_train.py
            activation="elu"
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Could not find model_state_dict in checkpoint")
    
    # Set to evaluation mode
    model.eval()
    
    # For ONNX export, we only need the actor part of the ActorCritic model
    # Create a wrapper class that only returns the actor output
    class ActorWrapper(torch.nn.Module):
        def __init__(self, actor_critic):
            super().__init__()
            self.actor = actor_critic.actor
            
        def forward(self, x):
            return self.actor(x)
    
    actor_model = ActorWrapper(model)
    
    # Create a dummy input tensor with shape [batch_size, obs_dim]
    # We'll use batch_size=1 for tracing, but it will be dynamic in the export
    dummy_input = torch.randn(1, obs_dim)
    
    # Export the model
    print(f"Exporting model to ONNX format at {onnx_path}")
    torch.onnx.export(
        actor_model,         # model being run (only the actor part)
        dummy_input,         # model input (or a tuple for multiple inputs)
        onnx_path,           # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,    # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],   # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model converted successfully to {onnx_path}")

def find_checkpoint(input_dir, checkpoint_num=None):
    """
    Find the checkpoint file to convert based on iteration number.
    
    Args:
        input_dir: Directory containing model checkpoints
        checkpoint_num: Specific checkpoint iteration number to find (if None, use highest)
        
    Returns:
        Path to the selected checkpoint file
    """
    # Find all model checkpoint files
    checkpoint_pattern = re.compile(r'model_(\d+)\.pt$')
    checkpoints = []
    
    for file in Path(input_dir).glob("model_*.pt"):
        match = checkpoint_pattern.search(file.name)
        if match:
            iter_num = int(match.group(1))
            checkpoints.append((iter_num, file))
    
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {input_dir}")
    
    # Sort checkpoints by iteration number
    checkpoints.sort(key=lambda x: x[0])
    
    if checkpoint_num is None:
        # Return the highest iteration checkpoint
        return checkpoints[-1][1]
    else:
        # Find the specific checkpoint
        for iter_num, file in checkpoints:
            if iter_num == checkpoint_num:
                return file
        
        # If we get here, the requested checkpoint wasn't found
        available_checkpoints = [str(iter_num) for iter_num, _ in checkpoints]
        raise ValueError(f"Checkpoint {checkpoint_num} not found. Available checkpoints: {', '.join(available_checkpoints)}")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PyTorch models")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save ONNX models")
    parser.add_argument("--obs_dim", type=int, default=NUM_OBSERVATION_FEATURES, 
                        help="Observation dimension (number of features in input vector)")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Specific checkpoint iteration number to convert (default: highest available)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Find the checkpoint to convert
        checkpoint_file = find_checkpoint(args.input_dir, args.checkpoint)
        
        # Create output path
        checkpoint_name = checkpoint_file.name
        onnx_path = Path(args.output_dir) / checkpoint_name.replace('.pt', '.onnx')
        
        # Create parent directories if they don't exist
        os.makedirs(onnx_path.parent, exist_ok=True)
        
        # Convert the model
        convert_pt_to_onnx(str(checkpoint_file), str(onnx_path), args.obs_dim)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()