import numpy as np
from utils.BoundaryCheck import BoundaryCheck
from utils.initialization import initialization

'''%%--------------萤火虫优化算法----------------------%%
%% 输入：
%   pop:种群数量
%   dim:单个萤火虫的维度
%   ub:萤火虫上边界信息，维度为[1,dim];
%   lb:萤火虫下边界信息，维度为[1,dim];
%   fobj:为适应度函数接口
%   maxIter: 算法的最大迭代次数，用于控制算法的停止。
%% 输出：
%   Best_Pos：为萤火虫算法找到的最优位置
%   Best_fitness: 最优位置对应的适应度值
%   IterCure:  用于记录每次迭代的最佳适应度，即后续用来绘制迭代曲线。
'''


def FA(pop, dim, ub, lb, fobj, maxIter, X,city_coordinates):
    beta0 = 2  # 最大吸引度
    gamma = 1  # 光强吸收系数
    alpha = 0.2  # 步长因子
    dmax = np.linalg.norm(ub - lb)  # 空间最大距离，用于距离归一化

    # 初始化种群位置
    # X = initialization(pop, ub, lb, dim)

    # 计算适应度值
    fitness = np.apply_along_axis(fobj, 1, X)
    # 初始化全局最优解
    gBestFitness = np.min(fitness)
    gBest = X[np.argmin(fitness), :]

    # 初始化迭代曲线记录
    IterCurve = np.zeros(maxIter)

    # 计算所有萤火虫之间的距离矩阵，避免重复计算
    distance_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2) / dmax
    # 计算所有萤火虫之间的吸引度矩阵
    beta_matrix = beta0 * np.exp(-gamma * distance_matrix ** 2)
    timeE=0
    for t in range(maxIter):
        print(t)
        for i in range(pop):
            for j in range(pop):
                if fitness[j] < fitness[i]:
                    # 萤火虫位置更新
                    attract = beta_matrix[i, j] * (X[j, :] - X[i, :])
                    random_step = alpha * (np.random.rand(dim) - 0.5) * (ub - lb)
                    X[i, :] += attract + random_step
                    X[i, :] = BoundaryCheck(X[i, :], ub, lb)  # 边界检查

            # 更新适应度
            new_fitness = fobj(X[i, :])
            if new_fitness < fitness[i]:  # 如果新适应度更好
                fitness[i] = new_fitness
                if new_fitness < gBestFitness:  # 如果新适应度是全局最优
                    timeE=t
                    gBestFitness = new_fitness
                    gBest = X[i, :].copy()  # 更新全局最优

        # 记录迭代曲线
        IterCurve[t] = gBestFitness
        # 更新吸引度矩阵，因为位置已经更新
        distance_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2) / dmax
        beta_matrix = beta0 * np.exp(-gamma * distance_matrix ** 2)

    return timeE,X, gBest, gBestFitness, IterCurve

# Example usage:
# Define your own fobj function that calculates the fitness of a solution
# Then call: best_pos, best_fitness, iter_curve, final_population = FA(pop, dim, ub, lb, fobj, max_iter, initial_population)
