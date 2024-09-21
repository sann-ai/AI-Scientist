import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os.path as osp

# LOAD FINAL RESULTS:
datasets = ["rotated_surface_code"]
folders = os.listdir("./")
final_results = {}
results_info = {}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        run_info = {}
        
        for dataset in datasets:
            run_info[dataset] = {}
            for seed_offset in range(len(results_dict)):
                key = f"{dataset}_{seed_offset}_final_info"
                if key in results_dict:
                    info = results_dict[key]
                    run_info[dataset][seed_offset] = {
                        "error_rates": info["error_rates"],
                        "detection_rates": info["detection_rates"]
                    }
        
        results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baselines",
}

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [plt.cm.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot: Error Detection Rate vs Error Rate for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        error_rates = results_info[run][dataset][0]["error_rates"]
        detection_rates = results_info[run][dataset][0]["detection_rates"]
        plt.plot(error_rates, detection_rates, label=labels[run], color=colors[i], marker='o')

    plt.title(f"Error Detection Rate vs Error Rate for {dataset}")
    plt.xlabel("Error Rate")
    plt.ylabel("Error Detection Rate")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"error_detection_rate_{dataset}.png")
    plt.close()

print("プロットが完了しました。")
