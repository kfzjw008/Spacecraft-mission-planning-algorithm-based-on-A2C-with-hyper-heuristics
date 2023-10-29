import numpy as np
from tqdm import tqdm

from utils.initialization import initialization
#树种优化算法
'''
%%--------------树种优化算法----------------------%%
%% 输入：
%   pop:个体数量
%   dim:单个个体的维度
%   ub:上边界信息，维度为[1,dim];
%   lb:下边界信息，维度为[1,dim];
%   fobj:为适应度函数接口
%   maxIter: 算法的最大迭代次数，用于控制算法的停止。
%% 输出：
%   Best_Pos：为树种算法找到的最优位置
%   Best_fitness: 最优位置对应的适应度值
%   IterCure:  用于记录每次迭代的最佳适应度，即后续用来绘制迭代曲线。
'''
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

    return trees,gBest, gBestFitness, IterCurve, trees






