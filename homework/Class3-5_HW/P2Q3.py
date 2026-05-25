import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_doe(bounds, num_samples):
    """
    使用拉丁超立方采样（LHS）生成DOE样本，并可选择评估它们。
    参数:
    bounds : np.ndarray
        一个形状为 (n_parameters, 2) 的二维数组，每行指定参数的下限和上限。
    num_samples : 要生成的DOE样本数量。
    返回:
    X : np.ndarray
        缩放到指定边界的DOE采样点。
    """
    num_params = bounds.shape[0]
  
    # 生成归一化的LHS样本（每个值在 [0, 1] 内）
    normalized_samples = lhs(num_params, samples=num_samples)
  
    # 将样本缩放到由 bounds 定义的实际搜索空间
    X = normalized_samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return X

# 定义参数搜索空间（每行：[下限, 上限]）
bounds = np.array([
    [30, 110],      # 温度 (°C)
    [10, 100],      # 反应时间 (min)
    [0.835, 4.175]  # 催化剂浓度 (mM)
])

# 确定参数数量和初始样本数量
num_params = bounds.shape[0]
num_initial_samples = 10

# 生成DOE样本点
X_train = generate_doe(bounds, num_initial_samples)

# ---- 3D 可视化采样点 ----
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制LHS采样点
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2],
                     c='red', s=100, marker='o', edgecolors='black', linewidth=1.5,
                     label='LHS Samples')

# 设置坐标轴标签
ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Reaction Time (min)', fontsize=12)
ax.set_zlabel('Catalyst Concentration (mM)', fontsize=12)
ax.set_title('Latin Hypercube Sampling (LHS) in 3D Parameter Space\n'
             f'({num_initial_samples} samples)', fontsize=14)

# 添加网格线
ax.grid(True, alpha=0.3)

# 添加样本点标签
for i, (x, y, z) in enumerate(X_train):
    ax.text(x, y, z, f'  S{i+1}', fontsize=9, color='darkred')

# 绘制边界框以显示搜索空间
# 定义边界框的顶点
verts = [
    [bounds[0, 0], bounds[1, 0], bounds[2, 0]],  # 0
    [bounds[0, 1], bounds[1, 0], bounds[2, 0]],  # 1
    [bounds[0, 1], bounds[1, 1], bounds[2, 0]],  # 2
    [bounds[0, 0], bounds[1, 1], bounds[2, 0]],  # 3
    [bounds[0, 0], bounds[1, 0], bounds[2, 1]],  # 4
    [bounds[0, 1], bounds[1, 0], bounds[2, 1]],  # 5
    [bounds[0, 1], bounds[1, 1], bounds[2, 1]],  # 6
    [bounds[0, 0], bounds[1, 1], bounds[2, 1]],  # 7
]

# 定义边界框的边
edges = [
    [verts[0], verts[1]], [verts[1], verts[2]], [verts[2], verts[3]], [verts[3], verts[0]],
    [verts[4], verts[5]], [verts[5], verts[6]], [verts[6], verts[7]], [verts[7], verts[4]],
    [verts[0], verts[4]], [verts[1], verts[5]], [verts[2], verts[6]], [verts[3], verts[7]]
]

# 绘制边界线
for edge in edges:
    ax.plot3D(*zip(*edge), color='blue', alpha=0.3, linewidth=1)

# 添加图例
ax.legend(loc='upper left')

# 调整视角
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('P3Q3_lhs_samples_3d.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("拉丁超立方采样 (LHS) 结果")
print("=" * 60)
print(f"\n参数搜索空间:")
print(f"  温度 (°C):         [{bounds[0, 0]:.1f}, {bounds[0, 1]:.1f}]")
print(f"  反应时间 (min):     [{bounds[1, 0]:.1f}, {bounds[1, 1]:.1f}]")
print(f"  催化剂浓度 (mM):    [{bounds[2, 0]:.3f}, {bounds[2, 1]:.3f}]")
print(f"\n生成的 {num_initial_samples} 个采样点:")
print("-" * 60)
print(f"{'Sample':<10} {'Temperature':<15} {'Time':<15} {'Concentration':<15}")
print("-" * 60)
for i, sample in enumerate(X_train):
    print(f"S{i+1:<9} {sample[0]:<15.2f} {sample[1]:<15.2f} {sample[2]:<15.4f}")
print("=" * 60)