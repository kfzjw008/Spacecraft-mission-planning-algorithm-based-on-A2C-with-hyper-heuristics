import numpy as np


from utils.BoundaryCheck import BoundaryCheck



'''%%--------------果蝇优化算法----------------------%%
%% 输入：
%   pop:种群数量
%   dim:单个果蝇的维度
%   ub:果蝇上边界信息，维度为[1,dim];
%   lb:果蝇下边界信息，维度为[1,dim];
%   fobj:为适应度函数接口
%   maxIter: 算法的最大迭代次数，用于控制算法的停止。
%% 输出：
%   Best_Pos：为果蝇算法找到的最优位置
%   Best_fitness: 最优位置对应的适应度值
%   IterCure:  用于记录每次迭代的最佳适应度，即后续用来绘制迭代曲线。'''


def FOA(pop, dim, ub, lb, fobj, maxIter, X,city_coordinates):
    # 初始化果蝇位置
    ub1 = np.ones(dim)
    lb1 = np.zeros(dim)
    # 如果X非空，则计算X的最优解作为基准点
    if X is not None:
        fitness_X = np.array([fobj(individual, city_coordinates) for individual in X])
        bestIndex = np.argmin(fitness_X)
        X_axis = X[bestIndex, :]
        Y_axis = X[bestIndex, :]
    else:
        # 如果X为空，则随机生成基准点
        X_axis = np.random.uniform(lb[0], ub[0], dim)
        Y_axis = np.random.uniform(lb[1], ub[1], dim)
    Best_fitness = float('inf')  # 初始化最佳适应度值
    #X = np.zeros((pop, dim))
    Y =X
    S = np.zeros((pop, dim))
    Dist = np.zeros((pop, dim))
    Smell = np.zeros(pop)
    IterCurve = np.zeros(maxIter)
    timeE=0
    for t in range(maxIter):
        if t==400:
            c=1
        for i in range(pop):
            # 果蝇通过嗅觉要寻找食物的任意方向
            X[i, :] = X_axis + (np.random.rand(dim) * 2 - 1)
            Y[i, :] = Y_axis + (np.random.rand(dim) * 2 - 1)
            Dist[i, :] = np.sqrt(X[i, :] ** 2 + Y[i, :] ** 2)  # 计算距离
            Temp = 1 / Dist[i, :]  # 计算距离的倒数
            S[i, :] = Temp * (ub - lb) + lb  # 等比例放大到空间
            S[i, :] = BoundaryCheck(S[i, :], ub, lb)  # 边界检查，防止越界
            Smell[i] = fobj(S[i, :],city_coordinates)  # 计算浓度值，即适应度值

        bestSmeall, bestindex = np.min(Smell), np.argmin(Smell)  # 寻找最佳浓度值以及对应的果蝇索引
        # 保留最优初始位置和初始味道浓度
        for i in range(pop):
            X_axis = X[bestindex, :]
            Y_axis = Y[bestindex, :]

        if bestSmeall < Best_fitness:
            timeE=t
            Best_fitness = bestSmeall
            Best_Pos = S[bestindex, :].copy()

        # 记录每次迭代最优值
        IterCurve[t] = Best_fitness
        #X[:, :] = S[bestindex, :]

    return timeE,X,Best_Pos, Best_fitness, IterCurve