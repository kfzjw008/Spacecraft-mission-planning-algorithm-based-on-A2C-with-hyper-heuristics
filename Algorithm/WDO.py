import numpy as np

from utils.BoundaryCheck import BoundaryCheck
from utils.initialization import initialization
'''
%%--------------风驱动优化算法----------------------%%
%% 输入：
%   pop:个体数量
%   dim:单个个体的维度
%   ub:上边界信息，维度为[1,dim];
%   lb:下边界信息，维度为[1,dim];
%   fobj:为适应度函数接口
%   maxIter: 算法的最大迭代次数，用于控制算法的停止。
%% 输出：
%   Best_Pos：为风驱动算法找到的最优位置
%   Best_fitness: 最优位置对应的适应度值
%   IterCure:  用于记录每次迭代的最佳适应度，即后续用来绘制迭代曲线。

'''

def WDO(pop, dim, ub, lb, fobj, maxIter, pos,city_coordinates):
    RT = 3  # RT系数
    g = 0.2  # 引力常数
    alp = 0.4  # 更新公式中的常量
    c = 0.4  # 科氏力影响
    maxV = 0.3 * ub  # 速度上边界
    minV = -0.3 * ub  # 速度下边界

    # 初始化速度
    V = np.random.uniform(minV, maxV, (pop, dim))

    # 计算压力值（适应度值）
    fitness = np.array([fobj(pos[i, :],city_coordinates) for i in range(pop)])

    # 寻找适应度最小的位置,记录全局最优值
    indexSort = np.argsort(fitness)
    gBest = pos[indexSort[0], :]
    gBestFitness = fitness[indexSort[0]]

    # 种群排序
    pos = pos[indexSort, :]
    fitness = fitness[indexSort]

    IterCurve = np.zeros(maxIter)
    timeE=0
    # 开始迭代
    for t in range(maxIter):
        for i in range(pop):
            # 随机选择维度
            a = np.random.permutation(dim)
            velot = V[i, a]

            # 更新速度
            # 添加一个条件以避免除以零
            if i == 0:
                div_factor = 1
            else:
                div_factor = i
            V[i, :] = (1 - alp) * V[i, :] - (g * pos[i, :]) + \
                      abs(1 - 1 / div_factor) * ((gBest - pos[i, :]) * RT) + \
                      (c * velot / div_factor)

            # 速度边界检查
            V[i, :] = BoundaryCheck(V[i, :], maxV, minV)

            # 更新空气粒子位置
            pos[i, :] = pos[i, :] + V[i, :]
            pos[i, :] = BoundaryCheck(pos[i, :], ub, lb)

            # 计算适应度值
            fitness[i] = fobj(pos[i, :],city_coordinates)
            if fitness[i] < gBestFitness:
                timeE=t
                gBestFitness = fitness[i]
                gBest = pos[i, :]

        indexSort = np.argsort(fitness)

        # 种群排序
        pos = pos[indexSort, :]
        V = V[indexSort, :]
        fitness = fitness[indexSort]

        IterCurve[t] = gBestFitness

    Best_Pos = gBest
    Best_fitness = gBestFitness

    return timeE,pos,Best_Pos, Best_fitness, IterCurve