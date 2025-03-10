import os
import subprocess
import numpy as np

# Hyperparameter grid
hyperparams = {
    'window_size': [5, 10, 15],
    'latent_dim': [16, 32, 64],
    'beta': [0.01, 0.05, 0.1],
    'num_runs': 3
}

# Base results directory
results_dir = "hyperparam_results"
os.makedirs(results_dir, exist_ok=True)

for ws in hyperparams['window_size']:
    for ld in hyperparams['latent_dim']:
        for beta in hyperparams['beta']:
            # Create a folder for this combination
            combo_dir = f"{results_dir}/ws{ws}_ld{ld}_beta{beta}"
            os.makedirs(combo_dir, exist_ok=True)

            for run in range(hyperparams['num_runs']):
                print(f"Running: ws={ws}, ld={ld}, beta={beta}, run={run}")

                # Create sub-folder for this specific run
                run_dir = f"{combo_dir}/run_{run}"
                os.makedirs(run_dir, exist_ok=True)

                # Construct training command
                cmd = f"""
                python train_vmp.py \
                    --window_size {ws} \
                    --latent_dim {ld} \
                    --beta {beta} \
                    --run {run} \
                    --weight_decay 1e-5 \
                    --lr 1e-4
                """

                # Run training and capture output
                result = subprocess.run(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffering
                    universal_newlines=True
                )

                # Save training logs in the run folder
                with open(f"{run_dir}/training.log", "w") as log_file:
                    for line in result.stdout.split('\n'):
                        print(line)  # Show in console
                        log_file.write(line + "\n")

                # If training was successful, move the loss file to the run folder
                if result.returncode == 0:
                    model_dir = f"models/ws{ws}_ld{ld}_beta{beta}"
                    loss_file_src = f"{model_dir}/train_losses.npy"
                    loss_file_dest = f"{run_dir}/train_losses.npy"

                    if os.path.exists(loss_file_src):
                        os.rename(loss_file_src, loss_file_dest)
