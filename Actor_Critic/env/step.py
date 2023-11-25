import numpy as np

from Algorithm.GWO import GWO
from Algorithm.PSO import PSO
from Algorithm.SCA import SCA
from Algorithm.TSA import TSA


def step(action, pop, dim, ub, lb, fun1, vmax, vmin, maxIter, Best_Pos, states):
    # 在这里根据传入的动作执行相应的函数，并计算奖励

    # 假设动作为0，选择执行PSO
    if action == 0:
        X, Best_Pos, Best_fitness, IterCurve = PSO(pop, dim, ub, lb, fun1, vmax, vmin, maxIter, states[0])
        reward = -Best_fitness  # 计算奖励，取相反数
    # 假设动作为1，选择执行GWO
    elif action == 1:
        X, Best_Pos, Best_fitness, IterCurve = GWO(pop, dim, ub, lb, fun1, maxIter, states[0])
        reward = -Best_fitness
    # 假设动作为2，选择执行SCA
    elif action == 2:
        X, Best_Pos, Best_fitness, IterCurve = SCA(pop, dim, ub, lb, fun1, maxIter, states[0])
        reward = -Best_fitness
    # 假设动作为3，选择执行TSA
    elif action == 3:
        X, Best_Pos, Best_fitness, IterCurve, trees = TSA(pop, dim, ub, lb, fun1, maxIter, states[0])
        reward = -Best_fitness
    else:
        print("action:"+str(action))
        raise ValueError("Invalid action!")

    # 假设下一个状态为随机生成的状态
    next_state = states  # 根据实际情况生成下一个状态
    next_state[0] = X

    # 假设每次都没有终止
    done = False

    return next_state, reward, done, Best_fitness, Best_Pos, {}
