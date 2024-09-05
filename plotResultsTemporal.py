import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def parse_results(file_path):
    results = defaultdict(dict)
    current_model = None
    current_dataset = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Identify model name (e.g., CLIP, DINOV2)
            if re.match(r'^[A-Z]+\w*:$', line):
                current_model = line[:-1]
            
            # Identify dataset (e.g., Downstream Evaluation on Chest X-Ray)
            elif "Downstream Evaluation on" in line:
                current_dataset = line.split(" on ")[1]
            
            # Identify metrics
            elif "{" in line and "}" in line:
                metrics = eval(line)
                if 'AUROC' in metrics:
                    results[current_dataset][current_model] = metrics['AUROC']
                else:
                    results[current_dataset][current_model] = metrics['AUROC']
    
    return results

def radar_factory(num_vars, frame='polygon'):
    # Compute angle for each axis
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_offset(np.pi / 2)
            self.set_theta_direction(-1)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta

def radar_plot(results, method_colors):
    # Get datasets and methods
    datasets = list(results.keys())
    methods = list(next(iter(results.values())).keys())
    
    num_vars = len(datasets)
    theta = radar_factory(num_vars)
    theta = theta + theta[:1]
    
    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))

    # Remove default radial labels
    # ax.yaxis.set_visible(False)  # Hide the default radial axis ticks and labels
    ax.set_yticklabels([])

    # Normalize each dataset axis
    dataset_ranges = {}
    for dataset in datasets:
        dataset_ranges[dataset] = {
            'min': 0.5 * min(results[dataset][method] for method in methods),
            'max': max(results[dataset][method] for method in methods)
        }
    
    # Plot each method
    for method, color in method_colors.items():
        values = [
            (results[dataset][method] - dataset_ranges[dataset]['min']) / 
            (dataset_ranges[dataset]['max'] - dataset_ranges[dataset]['min'])
            for dataset in datasets
        ]
        values = values + values[:1]
        ax.fill(theta, values, color=color, alpha=0.3, label=method)
        ax.plot(theta, values, color=color, linewidth=2)

    # Manually draw radial ticks and labels for each dataset
    for i, dataset in enumerate(datasets):
        min_val = dataset_ranges[dataset]['min']
        max_val = dataset_ranges[dataset]['max']
        ticks = np.linspace(0, 1, 5)
        tick_values = [min_val + tick * (max_val - min_val) for tick in ticks]

        for j, tick in enumerate(ticks):
            if j == 0:
                continue
            ax.text(theta[i], tick, f'{tick_values[j]:.2f}', color='black',  size=10, ha='center', weight='bold')
                    
    # Set axis labels
    ax.set_varlabels([d[:10] for d in datasets])
    
    # Push the axis labels away from the center
    ax.tick_params(pad=25)

    plt.savefig('results_exploreTime.png')
    
results = parse_results("results_temporal_exploration2.txt")
print(results)
method_colors = {
    'UniViT': 'red',
    'WITHOUT': 'blue',
}
datasets = [k for k in results.keys()]
results = {dataset: {method: results[dataset][method] for method in method_colors} for dataset in datasets}
ranks = {method: [] for method in method_colors}
for dataset in results:
    ordering = sorted(results[dataset], key=lambda x: results[dataset][x], reverse=True)
    for i, method in enumerate(ordering):
        ranks[method] += [i + 1]
for method in method_colors:
    print(f'{method}: {np.mean(ranks[method])}')
radar_plot(results, method_colors)