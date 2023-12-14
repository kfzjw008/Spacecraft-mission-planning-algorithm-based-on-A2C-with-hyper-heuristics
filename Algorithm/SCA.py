import numpy as np

from utils.BoundaryCheck import BoundaryCheck
from utils.initialization import initialization

#正余弦优化算法
'''
%%--------------正余弦优化算法----------------------%%
%% 输入：
%   pop:种群数量
%   dim:单个个体的维度
%   ub:上边界信息，维度为[1,dim];
%   lb:下边界信息，维度为[1,dim];
%   fobj:为适应度函数接口
%   maxIter: 算法的最大迭代次数，用于控制算法的停止。
%% 输出：
%   Best_Pos：为正余弦算法找到的最优位置
%   Best_fitness: 最优位置对应的适应度值
%   IterCure:  用于记录每次迭代的最佳适应度，即后续用来绘制迭代曲线。
'''
def SCA(pop, dim, ub, lb, fobj, maxIter, X,city_coordinates):
    a = 0.5

    if X is None:
        X = initialization(pop, ub, lb, dim)

    fitness = np.array([fobj(x,city_coordinates) for x in X])

    sorted_fitness = np.argsort(fitness)
    gBest = X[sorted_fitness[0]].copy()
    gBestFitness = fitness[sorted_fitness[0]]

    IterCurve = np.zeros(maxIter)
    timeE =0

    for t in range(maxIter):
        r1 = a - t * (a / maxIter)
        for i in range(pop):
            for j in range(dim):
                r2 = np.random.rand() * (2 * np.pi)
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()

                if r4 < 0.5:
                    X[i, j] = X[i, j] + (r1 * np.sin(r2) * np.abs(r3 * gBest[j] - X[i, j]))
                else:
                    X[i, j] = X[i, j] + (r1 * np.cos(r2) * np.abs(r3 * gBest[j] - X[i, j]))

            X[i] = BoundaryCheck(X[i], ub, lb)

        fitness = np.array([fobj(x,city_coordinates) for x in X])

        sorted_fitness = np.argsort(fitness)
        if fitness[sorted_fitness[0]] < gBestFitness:
            gBestFitness = fitness[sorted_fitness[0]]
            gBest = X[sorted_fitness[0]].copy()
            timeE = t

        IterCurve[t] = gBestFitness

    Best_Pos = gBest
    Best_fitness = gBestFitness

    return timeE,X,Best_Pos, Best_fitness, IterCurve

