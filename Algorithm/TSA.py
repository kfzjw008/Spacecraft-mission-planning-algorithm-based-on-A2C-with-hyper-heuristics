import numpy as np
from tqdm import tqdm

from utils.initialization import initialization


def TSA(pop, dim, ub, lb, fobj, maxIter, trees=None):
    low = int(0.1 * pop)
    high = int(0.25 * pop)
    ST = 0.1

    if trees is None:
        trees = initialization(pop, ub, lb, dim)

    fitness = np.zeros(pop)
    for i in range(pop):
        fitness[i] = fobj(trees[i])

    SortFitness = np.sort(fitness)
    indexSort = np.argsort(fitness)
    gBest = trees[indexSort[0]].copy()
    gBestFitness = SortFitness[0]

    IterCurve = np.zeros(maxIter)

    for t in tqdm(range(maxIter)):
        for i in range(pop):
            seedNum = np.random.randint(low, high + 1)
            seeds = np.zeros((seedNum, dim))
            obj_seeds = np.zeros(seedNum)

            minimum = np.min(fitness)
            min_indis = np.argmin(fitness)
            bestParams = trees[min_indis].copy()

            for j in range(seedNum):
                komsu = np.random.randint(0, pop)
                while i == komsu:
                    komsu = np.random.randint(0, pop)
                seeds[j] = trees[j].copy()

                for d in range(dim):
                    if np.random.rand() < ST:
                        seeds[j, d] = trees[i, d] + (bestParams[d] - trees[komsu, d]) * (np.random.rand() - 0.5) * 2
                    else:
                        seeds[j, d] = trees[i, d] + (trees[i, d] - trees[komsu, d]) * (np.random.rand() - 0.5) * 2

                    # 边界检查
                    if seeds[j, d] > ub[d]:
                        seeds[j, d] = ub[d]
                    elif seeds[j, d] < lb[d]:
                        seeds[j, d] = lb[d]

                    # 计算适应度值
                    obj_seeds[j] = fobj(seeds[j])

            mintohum = np.min(obj_seeds)
            mintohum_indis = np.argmin(obj_seeds)

            if mintohum < fitness[i]:
                trees[i] = seeds[mintohum_indis].copy()
                fitness[i] = mintohum

        min_tree = np.min(fitness)
        min_tree_index = np.argmin(fitness)

        if min_tree < gBestFitness:
            gBestFitness = min_tree
            gBest = trees[min_tree_index].copy()

        IterCurve[t] = gBestFitness

    return gBest, gBestFitness, IterCurve, trees






