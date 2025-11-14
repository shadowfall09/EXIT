import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取数据
with open('token_stats.json', 'r') as f:
    results = json.load(f)

datasets = ['2wikimultihop', 'HotpotQA', 'NQ', 'TQA']
dataset_labels = ['2WikiMultihop', 'HotpotQA', 'NQ', 'TriviaQA']

# 准备数据
official_0_1_compressed = [results['official_0.1'][d]['avg_compressed_tokens'] for d in datasets]
official_compressed = [results['official'][d]['avg_compressed_tokens'] for d in datasets]

# 计算压缩比
compression_ratio_0_1 = [results['official_0.1'][d]['avg_compressed_tokens'] / results['official_0.1'][d]['avg_original_tokens'] for d in datasets]
compression_ratio_official = [results['official'][d]['avg_compressed_tokens'] / results['official'][d]['avg_original_tokens'] for d in datasets]

# 创建图：压缩后的tokens柱状图 + 压缩比折线图 - 更宽更扁
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

x = range(len(datasets))
width = 0.35

# 柱状图 - 压缩后tokens
bars1 = ax1.bar([i - width/2 for i in x], official_compressed, width,
                label=r'Compressed - Public ($\tau = 0.5$)', alpha=0.8, color='#1f77b4')
bars2 = ax1.bar([i + width/2 for i in x], official_0_1_compressed, width,
                label=r'Compressed - Public ($\tau = 0.1$)', alpha=0.8, color='#ff7f0e')

ax1.set_xlabel('Dataset', fontsize=12, labelpad=-1)
ax1.set_ylabel('Average Compressed Tokens', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(dataset_labels, rotation=15, ha='right')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3, axis='y')

# 创建第二个y轴用于压缩比折线图
ax2 = ax1.twinx()

# 折线图 - 压缩比
line1 = ax2.plot(x, compression_ratio_official, marker='o',
                 label=r'Ratio - Public ($\tau = 0.5$)',
                 linewidth=2.5, markersize=8, color='#2ca02c')
line2 = ax2.plot(x, compression_ratio_0_1, marker='s',
                 label=r'Ratio - Public ($\tau = 0.1$)',
                 linewidth=2.5, markersize=8, color='#d62728')

ax2.set_ylabel('Compression Ratio', fontsize=12)
ax2.tick_params(axis='y', labelcolor='black')

# 合并图例 - 放在图表下方外侧,增加与xlabel的距离
bars = [bars1, bars2]
lines = line1 + line2
labels = [r'Compressed - Public ($\tau = 0.5$)', r'Compressed - Public ($\tau = 0.1$)',
          r'Ratio - Public ($\tau = 0.5$)', r'Ratio - Public ($\tau = 0.1$)']
ax1.legend(bars + lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), 
           ncol=2, fontsize=9, frameon=True)

plt.title('Compressed Tokens and Compression Ratio',
          fontsize=13, fontweight='bold', pad=15)
plt.subplots_adjust(bottom=0.28)
plt.savefig('token_compressed_combined.png', dpi=300, bbox_inches='tight')
print("图表已保存到 token_compressed_combined.png")
