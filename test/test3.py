import numpy as np
import matplotlib.pyplot as plt

def TSA(pop, dim, ub, lb, fobj, maxIter, trees=None):
    low = int(0.1 * pop)
    high = int(0.25 * pop)
    ST = 0.1

    if trees is None:
        trees = np.random.rand(pop, dim) * (ub - lb) + lb

    fitness = np.array([fobj(tree) for tree in trees])

    sorted_fitness = np.argsort(fitness)
    gBest = trees[sorted_fitness[0]]
    gBestFitness = fitness[sorted_fitness[0]]

    IterCurve = np.zeros(maxIter)

    for t in range(maxIter):
        for i in range(pop):
            seedNum = np.random.randint(low, high + 1)
            seeds = np.zeros((seedNum, dim))
            obj_seeds = np.zeros(seedNum)

            _, min_indis = np.min(fitness), np.argmin(fitness)
            best_params = trees[min_indis]

            for j in range(seedNum):
                komsu = np.random.randint(0, pop)
                while i == komsu:
                    komsu = np.random.randint(0, pop)

                seeds[j] = trees[j]

                for d in range(dim):
                    if np.random.rand() < ST:
                        seeds[j, d] = trees[i, d] + (best_params[d] - trees[komsu, d]) * (np.random.rand() - 0.5) * 2
                    else:
                        seeds[j, d] = trees[i, d] + (trees[i, d] - trees[komsu, d]) * (np.random.rand() - 0.5) * 2

                seeds[j] = np.clip(seeds[j], lb, ub)
                obj_seeds[j] = fobj(seeds[j])

            min_seed, min_seed_indis = np.min(obj_seeds), np.argmin(obj_seeds)

            if min_seed < fitness[i]:
                trees[i] = seeds[min_seed_indis]
                fitness[i] = min_seed

        min_tree, min_tree_index = np.min(fitness), np.argmin(fitness)

        if min_tree < gBestFitness:
            gBestFitness = min_tree
            gBest = trees[min_tree_index]

        IterCurve[t] = gBestFitness

    return gBest, gBestFitness, IterCurve, trees

# 示例使用的适应度函数
def example_fitness_function(x):
    return x[0]**2 + x[1]**2

# 设置参数
pop = 50
dim = 2
ub = 10
lb = -10
maxIter = 100

# 调用算法
Best_Pos, Best_fitness, IterCurve, trees = TSA(pop, dim, ub, lb, example_fitness_function, maxIter)

# 打印最优结果
print("最优位置:", Best_Pos)
print("最优适应度值:", Best_fitness)

# 绘制迭代曲线
plt.plot(range(maxIter), IterCurve)
plt.xlabel('迭代次数')
plt.ylabel('最优适应度值')
plt.title('迭代曲线')
plt.show()
