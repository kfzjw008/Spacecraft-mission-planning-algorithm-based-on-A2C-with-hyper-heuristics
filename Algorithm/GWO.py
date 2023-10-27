import numpy as np

from utils.BoundaryCheck import BoundaryCheck
from utils.initialization import initialization


def GWO(pop, dim, ub, lb, fobj, maxIter, Positions=None):
    # 定义Alpha，Beta，Delta狼的位置和适应度
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')

    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')

    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')

    # 初始化种群位置
    if Positions is None:
        Positions = initialization(pop, ub, lb, dim)

    # 计算适应度值
    fitness = np.array([fobj(pos) for pos in Positions])

    # 对适应度排序，找到Alpha，Beta，Delta狼
    sorted_fitness = np.argsort(fitness)
    Alpha_pos = Positions[sorted_fitness[0]].copy()
    Alpha_score = fitness[sorted_fitness[0]]

    Beta_pos = Positions[sorted_fitness[1]].copy()
    Beta_score = fitness[sorted_fitness[1]]

    Delta_pos = Positions[sorted_fitness[2]].copy()
    Delta_score = fitness[sorted_fitness[2]]

    gBest = Alpha_pos.copy()
    gBestFitness = Alpha_score

    IterCurve = np.zeros(maxIter)

    for t in range(maxIter):
        a = 2 - t * (2 / maxIter)

        for i in range(pop):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

            # 边界检查
            Positions[i] = BoundaryCheck(Positions[i], ub, lb)

        # 计算适应度值
        fitness = np.array([fobj(pos) for pos in Positions])

        # 更新 Alpha, Beta,  Delta狼
        sorted_fitness = np.argsort(fitness)
        if fitness[sorted_fitness[0]] < Alpha_score:
            Alpha_score = fitness[sorted_fitness[0]]
            Alpha_pos = Positions[sorted_fitness[0]].copy()

        if fitness[sorted_fitness[1]] < Beta_score:
            Beta_score = fitness[sorted_fitness[1]]
            Beta_pos = Positions[sorted_fitness[1]].copy()

        if fitness[sorted_fitness[2]] < Delta_score:
            Delta_score = fitness[sorted_fitness[2]]
            Delta_pos = Positions[sorted_fitness[2]].copy()

        gBest = Alpha_pos.copy()
        gBestFitness = Alpha_score

        IterCurve[t] = gBestFitness

    Best_Pos = gBest
    Best_fitness = gBestFitness

    return Best_Pos, Best_fitness, IterCurve, Positions
