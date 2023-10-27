import numpy as np

def initialization(pop, ub, lb, dim):
    """
    初始化粒子群

    参数：
        pop: 种群数量
        ub: 每个维度的变量上边界，维度为[1,dim]
        lb: 每个维度的变量下边界，维度为[1,dim]
        dim: 每个粒子的维度

    返回值：
        X: 输出的种群，维度[pop, dim]
    """
    X = np.zeros((pop, dim))  # 为X事先分配空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j] - lb[j]) * np.random.rand() + lb[j]  # 生成[lb,ub]之间的随机数
    return X

# 示例用法：
'''
pop = 10
ub = [5, 10, 15]
lb = [1, 2, 3]
dim = len(ub)

X = initialization(pop, ub, lb, dim)
print(X)
'''