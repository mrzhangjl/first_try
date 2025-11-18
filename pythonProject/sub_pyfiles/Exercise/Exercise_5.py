import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成 x 取值（从 -10 到 10，取 1000 个点，确保曲线平滑）
x = np.linspace(-10, 10, 1000)
y = sigmoid(x)

# 创建画布并绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='darkblue', linewidth=2.5)

# 添加辅助线和标注
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, label='y=0.5 (x=0时)')
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='black', linestyle='-', alpha=0.3)

# 设置坐标轴和标题
plt.xlabel('x', fontsize=12)
plt.ylabel('σ(x)', fontsize=12)
plt.title('Sigmoid Function Curve', fontsize=14)
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(alpha=0.2)

# 显示图像
plt.show()