# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import torch
import time
import matplotlib.pyplot as plt


def cpu_matrix_multiply(size):
    """使用NumPy在CPU上执行矩阵乘法"""
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)

    start_time = time.time()
    c = np.dot(a, b)
    end_time = time.time()

    return end_time - start_time


def gpu_matrix_multiply(size):
    """使用PyTorch在GPU上执行矩阵乘法"""
    if not torch.cuda.is_available():
        print("未检测到GPU，无法执行GPU矩阵乘法")
        return None

    a = torch.rand(size, size, dtype=torch.float32, device='cuda')
    b = torch.rand(size, size, dtype=torch.float32, device='cuda')

    # 预热运行，让GPU初始化
    torch.matmul(a, b)

    # 同步GPU以确保所有操作完成
    torch.cuda.synchronize()

    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # 等待计算完成
    end_time = time.time()

    return end_time - start_time


def run_performance_test(matrix_sizes):
    """运行性能测试并返回结果"""
    cpu_times = []
    gpu_times = []

    for size in matrix_sizes:
        print(f"测试矩阵大小: {size}x{size}")

        # 测试CPU性能
        cpu_time = cpu_matrix_multiply(size)
        cpu_times.append(cpu_time)
        print(f"CPU计算时间: {cpu_time:.6f} 秒")

        # 测试GPU性能
        gpu_time = gpu_matrix_multiply(size)
        if gpu_time is not None:
            gpu_times.append(gpu_time)
            print(f"GPU计算时间: {gpu_time:.6f} 秒")
            print(f"加速比: {cpu_time / gpu_time:.2f}x")
        else:
            gpu_times.append(None)
        print("-" * 50)

    return matrix_sizes, cpu_times, gpu_times


def plot_results(matrix_sizes, cpu_times, gpu_times):
    """绘制性能对比图"""
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, cpu_times, 'o-', label='CPU (NumPy)')

    # 过滤掉None值
    valid_gpu_indices = [i for i, t in enumerate(gpu_times) if t is not None]
    valid_gpu_sizes = [matrix_sizes[i] for i in valid_gpu_indices]
    valid_gpu_times = [gpu_times[i] for i in valid_gpu_indices]

    if valid_gpu_times:
        plt.plot(valid_gpu_sizes, valid_gpu_times, 's-', label='GPU (PyTorch)')

    plt.rcParams['font.sans-serif'] = ('Microsoft YaHei')
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('矩阵大小 (N x N)')
    plt.ylabel('计算时间 (秒)')
    plt.title('CPU与GPU矩阵乘法性能对比')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.show()


if __name__ == "__main__":
    # 设置要测试的矩阵大小
    matrix_sizes = [500, 1000, 2000, 4000, 8000 ]

    # 运行性能测试
    sizes, cpu_times, gpu_times = run_performance_test(matrix_sizes)

    # 绘制结果
    plot_results(sizes, cpu_times, gpu_times)