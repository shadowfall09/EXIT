import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
with open(r"outputs\exit_semantic_results\tiered_filter0.7_0.6\HotpotQA_results.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define the timing keys and their labels in chronological order
timing_components = [
    ('sentence_split', 'Sentence Split\n(SSC Step 1)'),
    ('query_encoding', 'Query Encoding\n(SSC Step 2)'),
    ('doc_encoding', 'Document Encoding\n(SSC Step 3)'),
    ('similarity_compute', 'Similarity\nComputation\n(SSC Step 4)'),
    ('filtering', 'Context\nRecombination\n(SSC Step 5)'),
    ('exit_inference', 'EXIT\nInference'),
    ('generation_time', 'Reader\nInference'),
]

# Extract timing data for each component
timing_data = {key: [] for key, _ in timing_components}

for result in data['results']:
    if 'timing' in result:
        timing = result['timing']
        for key, _ in timing_components:
            if key in timing:
                timing_data[key].append(timing[key])

# Calculate mean times for the stacked bar
mean_times = [np.mean(timing_data[key]) if timing_data[key] else 0 for key, _ in timing_components]

# Create figure with appropriate size for full-page width
fig, ax = plt.subplots(figsize=(16, 8))

# Define positions for box plots and the stacked bar
positions = np.arange(len(timing_components))
box_width = 0.6
bar_position = len(timing_components) + 0.5

# Define colors for each component (using a colorful palette)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']

# Create box and whisker plots
bp = ax.boxplot([timing_data[key] for key, _ in timing_components],
                 positions=positions,
                 widths=box_width,
                 patch_artist=True,
                 showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
                 medianprops=dict(color='black', linewidth=2),
                 boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                 whiskerprops=dict(color='black', linewidth=1.5),
                 capprops=dict(color='black', linewidth=1.5),
                 flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

# Color each box differently
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Create stacked bar chart for mean times
bottom = 0
bar_patches = []
for i, ((key, _), mean_time, color) in enumerate(zip(timing_components, mean_times, colors)):
    patch = ax.bar(bar_position, mean_time, width=box_width, bottom=bottom, 
                   color=color, edgecolor='black', linewidth=1.5, alpha=0.7)
    bar_patches.append(patch)
    
    # Add text label in the middle of each segment (skip query_encoding)
    if mean_time > 0.01 and key != 'query_encoding':  # Only label if segment is visible and not query_encoding
        ax.text(bar_position, bottom + mean_time/2, f'{mean_time:.3f}s', 
                ha='center', va='center', fontsize=9)
    
    bottom += mean_time

# Add total time at the top of the stacked bar
total_time = sum(mean_times)
ax.text(bar_position, bottom + 0.05, f'Total:\n{total_time:.3f}s', 
        ha='center', va='bottom', fontsize=10)

# Set x-axis labels
labels = [label for _, label in timing_components] + ['Full Pipeline\nAverage Timing']
ax.set_xticks(list(positions) + [bar_position])
ax.set_xticklabels(labels, fontsize=11)

# Set y-axis label
ax.set_ylabel('Time (seconds)', fontsize=13)

# Set title
ax.set_title('Inference Pipeline Timing Breakdown for HotpotQA', 
             fontsize=16, pad=20)

# Add grid for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Add a legend for box plot elements
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
               markeredgecolor='red', markersize=8, label='Mean'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

# Adjust layout
plt.tight_layout()

# Save figure
os.makedirs('outputs/plots', exist_ok=True)
plt.savefig('outputs/plots/timing_breakdown.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to outputs/plots/timing_breakdown.png")
print(f"\nSummary Statistics:")
print(f"Total samples: {len(data['results'])}")
print(f"Total average pipeline time: {total_time:.3f}s")
print(f"\nComponent average times:")
for (key, label), mean_time in zip(timing_components, mean_times):
    print(f"  {label.replace(chr(10), ' ')}: {mean_time:.3f}s ({mean_time/total_time*100:.1f}%)")
