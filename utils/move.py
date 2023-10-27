import numpy as np

def move(X, a, V):
    """
    引力搜索算法位置更新

    参数：
        X: 个体当前位置
        a: 个体加速
        V: 个体速度

    返回值：
        X: 更新后的位置
        V: 更新后的速度
    """
    N, dim = X.shape  # 获取种群维度
    V = np.random.rand(N, dim) * V + a  # 速度更新
    X = X + V  # 位置更新

    return X, V
'''
# 示例用法：
N = 5
dim = 3
X = np.random.rand(N, dim)
a = np.random.rand(N, dim)
V = np.random.rand(N, dim)

X, V = move(X, a, V)
print(X)
print(V)
'''