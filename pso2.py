# -*- coding: utf-8 -*-
import os

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


def PSOTSP(xy, dmat, Popsize, IterNum, showProg, showResult):
    N, _ = xy.shape
    nr, nc = dmat.shape

    if N != nr or N != nc:
        raise ValueError('城市坐标或距离矩阵输入有误')

    showProg = bool(showProg)
    showResult = bool(showResult)

    # 画出城市位置分布图
    plt.figure(1)
    plt.plot(xy[:, 0], xy[:, 1], 'k.', markersize=14)
    plt.title('城市坐标位置')

    # PSO参数初始化
    c1 = 0.1
    c2 = 0.075
    w = 1
    pop = np.zeros((Popsize, N), dtype=int)
    v = np.zeros((Popsize, N), dtype=int)
    iter = 1
    fitness = np.zeros(Popsize)
    Pbest = np.zeros((Popsize, N), dtype=int)
    Pbest_fitness = np.zeros(Popsize)
    Gbest = np.zeros((IterNum, N), dtype=int)
    Gbest_fitness = np.zeros(IterNum)
    Length_ave = np.zeros(IterNum)

    ws = 1
    we = 0.5

    # 产生初始位置和速度
    for i in range(Popsize):
        pop[i, :] = np.random.permutation(N)
        v[i, :] = np.random.permutation(N)

    # 计算粒子适应度值
    for i in range(Popsize):
        for j in range(N - 1):
            fitness[i] += dmat[pop[i, j], pop[i, j + 1]]
        fitness[i] += dmat[pop[i, -1], pop[i, 0]]

    Pbest_fitness = fitness.copy()
    Pbest = pop.copy()
    Gbest_fitness[0] = np.min(fitness)
    min_index = np.argmin(fitness)
    Gbest[0, :] = pop[min_index, :]
    Length_ave[0] = np.mean(fitness)

    # 迭代寻优
    while iter < IterNum:
        iter += 1
        w = ws - (ws - we) * (iter / IterNum) ** 2
        if( iter%100==0):
            print("maxiter="+str(IterNum)+" YourIter="+str(iter))

        # 个体极值序列交换部分
        change1 = positionChange(Pbest, pop)
        change1 = changeVelocity(c1, change1)

        # 群体极值序列交换部分
        change2 = positionChange(np.tile(Gbest[iter - 2, :], (Popsize, 1)), pop)
        change2 = changeVelocity(c2, change2)

        # 原速度部分
        v = OVelocity(w, v)

        # 修正速度
        for i in range(Popsize):
            for j in range(N):
                if change1[i, j] != 0:
                    v[i, j] = change1[i, j]
                if change2[i, j] != 0:
                    v[i, j] = change2[i, j]

        # 更新粒子位置
        pop = updatePosition(pop, v)

        # 适应度值更新
        fitness = np.zeros(Popsize)
        for i in range(Popsize):
            for j in range(N - 1):
                fitness[i] += dmat[pop[i, j], pop[i, j + 1]]
            fitness[i] += dmat[pop[i, -1], pop[i, 0]]

        # 个体极值与群体极值的更新
        for i in range(Popsize):
            if fitness[i] < Pbest_fitness[i]:
                Pbest_fitness[i] = fitness[i]
                Pbest[i, :] = pop[i, :]

        minvalue = np.min(fitness)
        if minvalue < Gbest_fitness[iter - 2]:
            Gbest_fitness[iter - 1] = minvalue
            min_index = np.argmin(fitness)
            Gbest[iter - 1, :] = pop[min_index, :]
            if showProg:
                plt.figure(2)
                ax = plt.gca()
                for i in range(N - 1):
                    plt.plot([xy[int(Gbest[iter, i]), 0], xy[int(Gbest[iter, i + 1]), 0]],
                             [xy[int(Gbest[iter, i]), 1], xy[int(Gbest[iter, i + 1]), 1]], 'bo-', linewidth=2)
                    ax.set_prop_cycle(None)  # 重置颜色循环
                plt.plot([xy[int(Gbest[iter, -1]), 0], xy[int(Gbest[iter, 0]), 0]],
                         [xy[int(Gbest[iter, -1]), 1], xy[int(Gbest[iter, 0]), 1]], 'bo-', linewidth=2)
                plt.title('最优路线距离 = {:.2f}，迭代次数 = {}次'.format(minvalue, iter))


        else:
            Gbest_fitness[iter - 1] = Gbest_fitness[iter - 2]
            Gbest[iter - 1, :] = Gbest[iter - 2, :]

        Length_ave[iter - 1] = np.mean(fitness)

    # 结果显示
    Shortest_Length = np.min(Gbest_fitness)
    index = np.argmin(Gbest_fitness)
    BestRoute = Gbest[index, :]

    if showResult:
        plt.figure(3)
        plt.plot(np.concatenate((xy[BestRoute, 0], [xy[BestRoute[0], 0]])),
                 np.concatenate((xy[BestRoute, 1], [xy[BestRoute[0], 1]])), 'o-')
        plt.grid(True)
        plt.xlabel('城市位置横坐标')
        plt.ylabel('城市位置纵坐标')
        plt.title('粒子群算法优化路径最短距离：{:.2f}'.format(Shortest_Length))

        plt.figure(4)
        plt.plot(range(1, IterNum + 1), Gbest_fitness[:IterNum], 'b',
                 range(1, IterNum + 1), Length_ave[:IterNum], 'r:')
        plt.legend(['最短距离', '平均距离'])
        plt.xlabel('迭代次数')
        plt.ylabel('距离')
        plt.title('各代最短距离与平均距离对比')
        plt.show()

    Psorout = {'BestRoute': BestRoute, 'Shortest_Length': Shortest_Length}

    return Psorout


def positionChange(best, pop):
    change = np.zeros_like(best)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            change[i, j] = np.where(pop[i, :] == best[i, j])[0]
            temp = pop[i, j]
            pop[i, j] = pop[i, change[i, j]]
            pop[i, change[i, j]] = temp

    return change


def changeVelocity(c, change):
    for i in range(change.shape[0]):
        for j in range(change.shape[1]):
            if np.random.rand() > c:
                change[i, j] = 0

    return change


def OVelocity(c, v):
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.random.rand() > c:
                v[i, j] = 0

    return v


def updatePosition(pop, v):
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            if v[i, j] != 0:
                temp = pop[i, j]
                pop[i, j] = pop[i, v[i, j]]
                pop[i, v[i, j]] = temp

    return pop



import numpy as np

# 生成随机城市坐标
#np.random.seed(0)
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100

# 计算距离矩阵
N = cities.shape[0]
a = np.meshgrid(range(N))
dmat = np.reshape(np.sqrt(np.sum((np.expand_dims(cities, axis=1) - np.expand_dims(cities, axis=0))**2, axis=2)), (N, N))

# 运行 PSOTSP 算法
result = PSOTSP(cities, dmat, Popsize=100, IterNum=1000, showProg=True, showResult=True)

print("最优路径:", result['BestRoute'])
print("最短距离:", result['Shortest_Length'])
