import numpy as np

def random_walk_around_antlion(dim, maxIter, lb, ub, antlion, t):
    """
    蚁狮算法的随机游走

    参数：
        dim：蚁狮维度
        maxIter: 最大迭代次数
        lb：蚁狮下边界维度为1*dim
        ub：蚁狮上边界维度为1*dim
        antlion: 蚁狮
        t: 当前迭代次数

    返回值：
        RWs: 游走后归一化的位置
    """
    I = 1
    w = 1
    if t > 0.1 * maxIter:
        w = 2
    if t > maxIter * 0.5:
        w = 3
    if t > maxIter * 0.75:
        w = 4
    if t > maxIter * 0.9:
        w = 5
    if t > maxIter * 0.9:
        w = 6
    I = 1 + 10 ** w * (t / maxIter)

    lb = lb / I
    ub = ub / I

    if np.random.rand() < 0.5:
        lb += antlion
    else:
        lb = -lb + antlion

    if np.random.rand() >= 0.5:
        ub += antlion
    else:
        ub = -ub + antlion

    RWs = np.zeros((maxIter, dim))

    for i in range(dim):
        X = np.cumsum(2 * (np.random.rand(maxIter) > 0.5) - 1)
        a = np.min(X)
        b = np.max(X)
        c = lb[i]
        d = ub[i]
        X_norm = ((X - a) * (d - c)) / (b - a) + c
        RWs[:, i] = X_norm

    return RWs
'''
# 示例用法：
dim = 3
maxIter = 100
lb = np.array([0, 0, 0])
ub = np.array([10, 10, 10])
antlion = np.array([5, 5, 5])
t = 50

RWs = random_walk_around_antlion(dim, maxIter, lb, ub, antlion, t)
print(RWs)
'''