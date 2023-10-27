import numpy as np
'''
%% 引力搜索算法计算加速度
%输入： M所有个体质量
%       X所有个体位置
%       iteration 当前迭代次数
%       max_it最大迭代次数
%输出： a加速度
'''
def Acceleration(M, X, G, iteration, max_it):
    N, dim = X.shape  # 获取种群维度
    final_per = 2  # 在最后一次迭代时，只有两个个体相互吸引.
    kbest = final_per + (1 - iteration / max_it) * (100 - final_per)  # 计算kbest的数量
    kbest = round(N * kbest / 100)  # 计算kbest的数量
    Ms = np.argsort(-M)  # 对质量排序
    F = np.zeros((N, dim))

    for i in range(N):
        for ii in range(kbest):
            j = Ms[ii]
            if j != i:
                R = np.linalg.norm(X[i] - X[j], 2)  # 计算欧式距离
                for k in range(dim):
                    F[i, k] += np.random.rand() * M[j] * ((X[j, k] - X[i, k]) / (R + np.finfo(float).eps))  # 计算吸引力

    # 计算加速度
    a = F * G
    return a
'''
# 示例用法：
M = np.array([1, 2, 3, 4, 5])
X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
G = 0.5
iteration = 1
max_it = 10

a = Acceleration(M, X, G, iteration, max_it)
print(a)
'''