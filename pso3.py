import numpy as np
import matplotlib.pyplot as plt

def PSO_TSP(xy, dmat, Popsize, IterNum):
    N, _ = xy.shape

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

    # 迭代寻优
    while iter < IterNum:
        iter += 1
        w = ws - (ws - we) * (iter / IterNum) ** 2

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

        else:
            Gbest_fitness[iter - 1] = Gbest_fitness[iter - 2]
            Gbest[iter - 1, :] = Gbest[iter - 2, :]

    # 结果显示
    Shortest_Length = np.min(Gbest_fitness)
    index = np.argmin(Gbest_fitness)
    BestRoute = Gbest[index, :]

    return {'BestRoute': BestRoute, 'Shortest_Length': Shortest_Length}

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

def calculate_total_distance(sequence, dmat):
    total_distance = 0
    N = len(sequence)
    for i in range(N - 1):
        total_distance += dmat[sequence[i], sequence[i + 1]]
    total_distance += dmat[sequence[-1], sequence[0]]  # 回到起点
    return total_distance

def run_PSO_TSP(cities, dmat, Popsize, IterNum):
    result = PSO_TSP(cities, dmat, Popsize, IterNum)
    return {
        "BestRoute": result['BestRoute'],
        "Shortest_Length": result['Shortest_Length']
    }

# 生成随机城市坐标
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100

# 计算距离矩阵
N = cities.shape[0]
a = np.meshgrid(range(N))
dmat = np.reshape(np.sqrt(np.sum((np.expand_dims(cities, axis=1) - np.expand_dims(cities, axis=0))**2, axis=2)), (N, N))

# 运行 PSOTSP 算法
result = run_PSO_TSP(cities, dmat, Popsize=100, IterNum=1000)

print("最优路径:", result['BestRoute'])
print("最短距离:", result['Shortest_Length'])
