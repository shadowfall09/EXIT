import json
import matplotlib.pyplot as plt
import os

# Define the data files and their corresponding semantic filter relevance ratios
data_files = [
    (0.3, r"outputs\exit_semantic_results\tiered_filter0.7_0.3\HotpotQA_results.json"),
    (0.4, r"outputs\exit_semantic_results\tiered_filter0.7_0.4\HotpotQA_results.json"),
    (0.5, r"outputs\exit_semantic_results\tiered_filter0.7_0.5\HotpotQA_results.json"),
    (0.6, r"outputs\exit_semantic_results\tiered_filter0.7_0.6\HotpotQA_results.json"),
    (0.7, r"outputs\exit_semantic_results\filter70_th0.1\HotpotQA_results.json"),
]

# Initialize lists to store metrics
ratios = []
exact_match = []
f1_scores = []
latency = []
token_compression = []

# Read data from each file
for ratio, filepath in data_files:
    with open(filepath, 'r', encoding="utf-8") as f:
        data = json.load(f)
        metrics = data['metrics']
        
        ratios.append(ratio)
        exact_match.append(metrics['exact_match'])
        f1_scores.append(metrics['f1'])
        latency.append(metrics['latency'])
        token_compression.append(metrics['token_compression_ratio'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: QA Performance
color1 = 'tab:blue'
ax1.set_xlabel('Semantic Filter Relevance Ratio', fontsize=12)
ax1.set_ylabel('Exact Match', color=color1, fontsize=12)
line1 = ax1.plot(ratios, exact_match, 'o-', color=color1, label='Exact Match', linewidth=2, markersize=8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
color2 = 'tab:orange'
ax1_twin.set_ylabel('F1 Score', color=color2, fontsize=12)
line2 = ax1_twin.plot(ratios, f1_scores, 's-', color=color2, label='F1 Score', linewidth=2, markersize=8)
ax1_twin.tick_params(axis='y', labelcolor=color2)

ax1.set_title('QA Performance vs Semantic Filter Relevance Ratio', fontsize=14)
lines1 = line1 + line2
labels1 = [l.get_label() for l in lines1]
ax1.legend(lines1, labels1, loc='best')

# Right plot: Compression Performance
color3 = 'tab:green'
ax2.set_xlabel('Semantic Filter Relevance Ratio', fontsize=12)
ax2.set_ylabel('Latency (s)', color=color3, fontsize=12)
line3 = ax2.plot(ratios, latency, 'o-', color=color3, label='Latency', linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.grid(True, alpha=0.3)

ax2_twin = ax2.twinx()
color4 = 'tab:red'
ax2_twin.set_ylabel('Token Compression Ratio', color=color4, fontsize=12)
line4 = ax2_twin.plot(ratios, token_compression, 's-', color=color4, label='Token Compression Ratio', linewidth=2, markersize=8)
ax2_twin.tick_params(axis='y', labelcolor=color4)

ax2.set_title('Compression Performance vs Semantic Filter Relevance Ratio', fontsize=14)
lines2 = line3 + line4
labels2 = [l.get_label() for l in lines2]
ax2.legend(lines2, labels2, loc='best')

# Adjust layout and save
plt.tight_layout()
os.makedirs('outputs/plots', exist_ok=True)
plt.savefig('outputs/plots/tiered_tradeoff.png', dpi=300, bbox_inches='tight')
print("Plot saved to outputs/plots/tiered_tradeoff.png")
