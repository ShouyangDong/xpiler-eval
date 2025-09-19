import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. 数据定义
# -------------------------------
np.random.seed(42)
operators = [
    'Add', 'Relu', 'Sigmoid', 'Tanh', 'Sum', 'Mean', 'Max', 'Min',
    'Gemm', 'GemV', 'Bmm',
    'Conv1D', 'Conv2dNHWC', 'Conv2dNCHW', 'DepthwiseConv',
    'BatchNorm', 'LayerNorm', 'RMSNorm',
    'ReLU', 'GeLU', 'Swish', 'Softmax',
    'MaxPool2d', 'AvgPool2d', 'MinPool2d', 'SumPool2d',
    'Reshape', 'Transpose',
    'SelfAtten', 'DAT'
]

# 模拟数据
transpilers = {
    'C with VNNI → CUDA C':     {'perf_ratio': 0.88, 'corrected_cases': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]},
    'CUDA C → BANG C':     {'perf_ratio': 0.78, 'corrected_cases': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1]},
    'CUDA C → HIP':        {'perf_ratio': 0.90, 'corrected_cases': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]},
    'CUDA C → C with VNNI':     {'perf_ratio': 0.72, 'corrected_cases': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0]},
}

# -------------------------------
# 2. 设置绘图
# -------------------------------
n_ops = len(operators)
x_pos = np.arange(n_ops + 1)  # +1 for "Overall"
width = 0.35

fig, axes = plt.subplots(4, 1, figsize=(20, 14))
axes = axes.flatten()

# 颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 图例控制
legend_handles, legend_labels = [], []
first_legend = True
prefix_name = ["(a)", "(b)", "(c)", "(d)"]
# 绘制每个 transpiler
for idx, (name, data) in enumerate(transpilers.items()):
    ax = axes[idx]
    ax2 = ax.twinx()  # 双Y轴：左=性能，右=corrected cases
    
    # 主Y轴：性能柱状图
    pytorch_perf = [1.0] * n_ops
    xpiler_perf = [data['perf_ratio']] * n_ops
    overall_perf = np.mean(xpiler_perf)

    bar1 = ax.bar(x_pos[:-1] - width/2, pytorch_perf, width, color="gray", alpha=0.8, label='PyTorch (Baseline)')
    bar2 = ax.bar(x_pos[:-1] + width/2, xpiler_perf, width, color=colors[0], alpha=0.9, label='QiMeng-Xpiler')
    bar3 = ax.bar(x_pos[-1], overall_perf, width*1.2, color=colors[1], alpha=0.9, label='Overall Avg Perf')

    # 次Y轴：Corrected Cases 折线
    line, = ax2.plot(x_pos[:-1], data['corrected_cases'],
                     marker='o', color=colors[2], linewidth=2.5,
                     markersize=6, zorder=5, label='Corrected Cases (per op)')

    # 次Y轴：Overall Corrected Cases 点
    overall_correct = sum(data['corrected_cases']) / 30
    scatter = ax2.scatter(x_pos[-1], overall_correct, s=100, c=colors[2], marker='o',
                          edgecolors='black', linewidth=1.5, zorder=6,
                          label='Overall Avg Corrected Cases' if first_legend else "")
    ax2.grid(True, linestyle='--', alpha=0.6, linewidth=1.0)
    ax2.set_axisbelow(False)
    # 坐标轴设置
    ax.set_ylim(0, 1.3)
    ax2.set_ylim(0, 8.5)  # corrected cases: 0~8
    ax.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    name = prefix_name[idx] + " " + name
    ax.set_title(f"{name}", fontsize=14, fontweight='bold')

    # X轴标签：只在最下面一个图显示
    if idx == len(axes) - 1:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(operators + ['Overall'], rotation=60, ha='right')
    else:
        ax.set_xticklabels([])

    # 收集图例
    if first_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend_handles = handles + handles2
        legend_labels = labels + labels2
        first_legend = False

# 在图形顶部添加全局Y轴标签
fig.text(0.08, 0.5, 'Normalized Performance', va='center', rotation='vertical', fontsize=12)
fig.text(0.90, 0.5, 'Corrected Cases (0-8)', va='center', rotation=270, fontsize=12, color='darkgreen')

# 统一图例（最上方）
fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=5, fontsize=12)

# 布局调整
plt.subplots_adjust(top=0.90, bottom=0.18, left=0.12, right=0.88, hspace=0.2)
plt.savefig('transpiler_comparison.pdf', 
            format='pdf', 
            bbox_inches='tight',   # 自动裁剪白边
            pad_inches=0.05,       # 内边距，可设为0或0.05避免内容被切
            dpi=300) 
plt.show()
