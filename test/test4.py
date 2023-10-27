import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(solution, cities):
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += calculate_distance(cities[solution[i]], cities[solution[i + 1]])
    total_distance += calculate_distance(cities[solution[-1]], cities[solution[0]])
    return 1 / total_distance


def generate_random_solution(cities):
    return np.random.permutation(len(cities))


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


def mutate(solution):
    index1, index2 = np.random.choice(len(solution), 2, replace=False)
    solution[index1], solution[index2] = solution[index2], solution[index1]
    return solution


def TSA_TSP(pop, cities, maxIter):
    trees = []
    best_pos = None
    best_fitness = float('-inf')
    iter_curve = []

    for _ in range(pop):
        solution = generate_random_solution(cities)
        fitness = calculate_fitness(solution, cities)
        trees.append((solution, fitness))

    for iteration in range(maxIter):
        new_trees = []
        for i in range(pop):
            parent1, parent2 = trees[np.random.choice(pop, 2, replace=False)]
            child = crossover(parent1[0], parent2[0])
            if np.random.rand() < 0.1:
                child = mutate(child)
            fitness = calculate_fitness(child, cities)
            new_trees.append((child, fitness))

        trees += new_trees
        trees = sorted(trees, key=lambda x: x[1], reverse=True)[:pop]

        best_iter_fitness = trees[0][1]
        if best_iter_fitness > best_fitness:
            best_pos = trees[0][0]
            best_fitness = best_iter_fitness
            iter_curve.append(best_fitness)

    return best_pos, best_fitness, iter_curve, trees


# 示例用法
cities = np.array([[0, 0], [1, 3], [5, 5], [8, 0], [3, 8], [9, 9]])
pop = 10
maxIter = 1000

best_pos, best_fitness, iter_curve, trees = TSA_TSP(pop, cities, maxIter)
print("最优解:", best_pos)
print("最优适应度:", best_fitness)

# 绘制迭代曲线
plt.plot(iter_curve)
plt.xlabel('迭代次数')
plt.ylabel('适应度')
plt.title('适应度迭代曲线')
plt.show()

# 打印结果
print("老吕牛逼！集成室牛逼！")
