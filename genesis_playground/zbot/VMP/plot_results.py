import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re

def extract_losses_from_log(log_path):
    """Extracts loss values from training.log"""
    losses = []
    pattern = re.compile(r"Epoch \d+/1000 \| Loss: ([\d\.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    loss_value = float(match.group(1))  # Extract numeric loss
                    losses.append(loss_value)
                except ValueError:
                    continue  # Ignore if parsing fails

    return np.array(losses)


def load_results(base_dir):
    results = {}
    
    # Iterate over each hyperparameter combination folder
    for combo_dir in os.listdir(base_dir):
        combo_path = os.path.join(base_dir, combo_dir)
        if not os.path.isdir(combo_path) or not combo_dir.startswith('ws'):
            continue

        # Extract hyperparameters from folder name
        params = combo_dir.split('_')
        ws = int(params[0][2:])      # "wsX"
        ld = int(params[1][2:])      # "ldX"
        beta = float(params[2][4:])  # "betaX.X"

        key = (ws, ld, beta)
        if key not in results:
            results[key] = []

        found_loss = False  # Track if any loss file is found

        # Iterate over run folders (run_0, run_1, etc.)
        for run_folder in sorted(os.listdir(combo_path)):  # Sorting ensures consistent order
            run_path = os.path.join(combo_path, run_folder)
            if not run_folder.startswith("run_") or not os.path.isdir(run_path):
                continue

            log_file = os.path.join(run_path, "training.log")
            if os.path.exists(log_file):
                #print(f"✅ Extracting losses from: {log_file}")  # Debugging output
                losses = extract_losses_from_log(log_file)
                if losses.size > 0:
                    results[key].append(losses)
                    found_loss = True
            else:
                print(f"⚠️ Missing log file: {log_file}")  # Debugging output
        
        if not found_loss:
            print(f"🚨 No valid loss data found for: {combo_dir}")

    return results




def plot_separate_comparisons(results):
    #plt.style.use("seaborn-whitegrid")

    # Group by ws, ld, and beta
    ws_values = sorted(set(k[0] for k in results.keys()))
    ld_values = sorted(set(k[1] for k in results.keys()))
    beta_values = sorted(set(k[2] for k in results.keys()))

    def plot_grouped(results, group_by_idx, title_prefix, filename_prefix):
        """Helper function to create plots grouped by a specific hyperparameter index"""
        for group_val in sorted(set(k[group_by_idx] for k in results.keys())):
            plt.figure(figsize=(12, 8))
            for key in results:
                if key[group_by_idx] == group_val:
                    ws, ld, beta = key
                    all_runs = np.array(results[key], dtype=object)

                    if len(all_runs) == 0:  
                        print(f"⚠️ Skipping {key}, no valid loss data found.")
                        continue
                    
                    max_len = max(len(run) for run in all_runs)
                    padded = np.array([np.pad(run, (0, max_len - len(run)), 
                                              mode='constant', constant_values=np.nan) 
                                       for run in all_runs], dtype=np.float64)

                    if np.isnan(padded).all():
                        print(f"⚠️ Skipping {key}, all values are NaN.")
                        continue

                    mean_loss = np.nanmean(padded, axis=0)
                    std_loss = np.nanstd(padded, axis=0)

                    label = f"LD={ld}, β={beta}, WS={ws}"
                    plt.plot(mean_loss, label=label)
                    plt.fill_between(range(len(mean_loss)), 
                                     mean_loss - std_loss,
                                     mean_loss + std_loss,
                                     alpha=0.2)

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{title_prefix} {group_val}")
            plt.legend()
            plt.savefig(f"{filename_prefix}_{group_val}.png")
            plt.close()

    plot_grouped(results, group_by_idx=0, title_prefix="Loss Curves for WS =", filename_prefix="ws_comparison")
    plot_grouped(results, group_by_idx=1, title_prefix="Loss Curves for LD =", filename_prefix="ld_comparison")
    plot_grouped(results, group_by_idx=2, title_prefix="Loss Curves for β =", filename_prefix="beta_comparison")

if __name__ == "__main__":
    results = load_results("hyperparam_results")
    plot_separate_comparisons(results)
