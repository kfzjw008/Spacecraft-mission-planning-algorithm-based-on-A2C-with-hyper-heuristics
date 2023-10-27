import numpy as np

from utils.BoundaryCheck import BoundaryCheck


def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def tsp_fobj(sequence, cities):
    total_distance = 0
    for i in range(len(sequence) - 1):
        total_distance += calculate_distance(cities[sequence[i]], cities[sequence[i + 1]])
    total_distance += calculate_distance(cities[sequence[-1]], cities[sequence[0]])
    return total_distance


def pso_tsp(pop, dim, ub, lb, cities, vmax, vmin, maxIter, X=None, V=None):
    c1 = 2.0
    c2 = 2.0

    if X is None:
        X = np.array([np.random.permutation(dim) for _ in range(pop)])


    if V is None:
        V = np.random.uniform(vmin, vmax, (pop, dim))

    fitness = np.array([tsp_fobj(x, cities) for x in X])

    pBest = X.copy()
    pBestFitness = fitness.copy()

    gBestIndex = np.argmin(fitness)
    gBestFitness = fitness[gBestIndex]
    gBest = X[gBestIndex].copy()

    Xnew = X.copy()
    fitnessNew = fitness.copy()

    IterCurve = np.zeros(maxIter)

    for t in range(maxIter):
        print("____________________")
        for i in range(pop):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            print(Xnew[i])
            V[i] = V[i] + c1 * r1 * (pBest[i] - X[i]) + c2 * r2 * (gBest - X[i])
            V[i] = BoundaryCheck(V[i], vmax, vmin)
            Xnew[i] = X[i] + V[i]

            print(Xnew[i])

            Xnew[i] = BoundaryCheck(Xnew[i], ub, lb)
            fitnessNew[i] = tsp_fobj(Xnew[i], cities)

            if fitnessNew[i] < pBestFitness[i]:
                pBest[i] = Xnew[i].copy()
                pBestFitness[i] = fitnessNew[i]

                if fitnessNew[i] < gBestFitness:
                    gBestFitness = fitnessNew[i]
                    gBest = Xnew[i].copy()

        X = Xnew.copy()
        fitness = fitnessNew.copy()

        Best_Pos = gBest
        Best_fitness = gBestFitness
        IterCurve[t] = gBestFitness

    return Best_Pos, Best_fitness, IterCurve

def plot_tsp_route(cities, route):
    plt.figure(figsize=(8, 8))
    plt.plot(cities[:, 0], cities[:, 1], 'k.', markersize=10)
    route = np.concatenate((route, [route[0]]))  # 添加回起点以闭合路线
    plt.plot(cities[route, 0], cities[route, 1], 'r-')
    plt.xlabel('城市位置横坐标')
    plt.ylabel('城市位置纵坐标')
    plt.title('TSP 最终迭代路线')
    plt.grid(True)
    plt.show()




# 示例使用
num_cities = 20
cities = np.random.rand(num_cities, 2) * 100

pop = 30
dim = num_cities
ub = np.arange(dim)
lb = np.zeros(dim)
vmax = (ub - lb) * 0.1
vmin = -vmax
maxIter = 10
print("城市坐标:")
print(cities)
print("\n最优路径:")
#print(Best_Pos)

Best_Pos, Best_fitness, IterCurve = pso_tsp(pop, dim, ub, lb, cities, vmax, vmin, maxIter)

print("最优序列:", Best_Pos)
print("最优适应度值:", Best_fitness)
print("迭代曲线:", IterCurve)

import matplotlib.pyplot as plt


# 示例使用
plot_tsp_route(cities, Best_Pos)
