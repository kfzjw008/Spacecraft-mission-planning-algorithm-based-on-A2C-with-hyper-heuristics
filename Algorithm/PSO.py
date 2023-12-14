import numpy as np

from utils.BoundaryCheck import BoundaryCheck



'''
%%--------------粒子群函数----------------------%%
%% 输入：
%   pop:种群数量
%   dim:单个粒子的维度
%   ub:粒子上边界信息，维度为[1,dim];
%   lb:粒子下边界信息，维度为[1,dim];
%   fobj:为适应度函数接口
%   vmax: 为速度的上边界信息，维度为[1,dim];
%   vmin: 为速度的下边界信息，维度为[1,dim];
%   maxIter: 算法的最大迭代次数，用于控制算法的停止。
%% 输出：
%   Best_Pos：为粒子群找到的最优位置
%   Best_fitness: 最优位置对应的适应度值
%   IterCure:  用于记录每次迭代的最佳适应度，即后续用来绘制迭代曲线。
'''

def PSO(pop, dim, ub, lb, fobj, vmax, vmin, maxIter, X, city_coordinates):
    c1 = 2.0
    c2 = 1.0

    if X is None:
        X = np.random.uniform(lb, ub, (pop, dim))


    V = np.random.uniform(vmin, vmax, (pop, dim))

    fitness = np.array([fobj(x,city_coordinates) for x in X])

    pBest = X.copy()
    pBestFitness = fitness.copy()

    gBestIndex = np.argmin(fitness)
    gBestFitness = fitness[gBestIndex]
    gBest = X[gBestIndex].copy()

    Xnew = X.copy()
    fitnessNew = fitness.copy()

    IterCurve = np.zeros(maxIter)
    timeE = 0
    for t in range(maxIter):
       # print(t)
        for i in range(pop):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            V[i] = V[i] + c1 * r1 * (pBest[i] - X[i]) + c2 * r2 * (gBest - X[i])
            V[i] = BoundaryCheck(V[i], vmax, vmin)
            Xnew[i] = X[i] + V[i]
            Xnew[i] = BoundaryCheck(Xnew[i], ub, lb)
            fitnessNew[i] = fobj(Xnew[i],city_coordinates)

            if fitnessNew[i] < pBestFitness[i]:
                pBest[i] = Xnew[i].copy()
                pBestFitness[i] = fitnessNew[i]

                if fitnessNew[i] < gBestFitness:
                    gBestFitness = fitnessNew[i]
                    timeE=t
                    gBest = Xnew[i].copy()

        X = Xnew.copy()
        fitness = fitnessNew.copy()

        Best_Pos = gBest
        Best_fitness = gBestFitness
        IterCurve[t] = gBestFitness

    return timeE,X,Best_Pos, Best_fitness, IterCurve


