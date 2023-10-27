import numpy as np

from utils.BoundaryCheck import BoundaryCheck
from utils.initialization import initialization


def SCA(pop, dim, ub, lb, fobj, maxIter, X=None):
    a = 2

    if X is None:
        X = initialization(pop, ub, lb, dim)

    fitness = np.array([fobj(x) for x in X])

    sorted_fitness = np.argsort(fitness)
    gBest = X[sorted_fitness[0]].copy()
    gBestFitness = fitness[sorted_fitness[0]]

    IterCurve = np.zeros(maxIter)

    for t in range(maxIter):
        r1 = a - t * (a / maxIter)
        for i in range(pop):
            for j in range(dim):
                r2 = np.random.rand() * (2 * np.pi)
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()

                if r4 < 0.5:
                    X[i, j] = X[i, j] + (r1 * np.sin(r2) * np.abs(r3 * gBest[j] - X[i, j]))
                else:
                    X[i, j] = X[i, j] + (r1 * np.cos(r2) * np.abs(r3 * gBest[j] - X[i, j]))

            X[i] = BoundaryCheck(X[i], ub, lb)

        fitness = np.array([fobj(x) for x in X])

        sorted_fitness = np.argsort(fitness)
        if fitness[sorted_fitness[0]] < gBestFitness:
            gBestFitness = fitness[sorted_fitness[0]]
            gBest = X[sorted_fitness[0]].copy()

        IterCurve[t] = gBestFitness

    Best_Pos = gBest
    Best_fitness = gBestFitness

    return Best_Pos, Best_fitness, IterCurve, X

