import time
from math import inf

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm, _tqdm

from Actor_Critic.env.step import step
from Algorithm.GWO import GWO
from Algorithm.PSO import PSO
from Algorithm.SCA import SCA
from utils.TSPGenerate import generate_tsp_coordinates
from utils.compare_numbers import compare_numbers, calculate_difference
# from test.test1 import YBest_Pos
# from test.test2ac import train
from utils.pltdraw import plot_city_coordinates_line


def train_on_policy_agent(EmaxIter, pop, dim, ub, lb, fun1, vmax, vmin, maxIter, X, agent, num_episodes,
                          city_coordinates):
    global current_time
    writer = SummaryWriter('../runs')

    # print("open!!!")
    return_list = []
    minfitness = inf
    with tqdm(total=num_episodes) as pbar:
        for i in range(num_episodes):
            agent.current_episode = i
            # ... do some work ...
            # print("train" + " " + str(i_episode))
            episode_return = 0
            city_coordinates = generate_tsp_coordinates(dim, 0, 100)
            transition_dict = {'statess': [], 'actions': [], 'next_states': [], 'rewards': [], 'cost': [], 'dones': [],
                               'Best_fitnesss': []}
            state = X  # 生成初始解
            first_row = X
            city_coordinates = np.array(city_coordinates)
            specific_column = city_coordinates[:, 1]
            second_row = np.tile(specific_column, (pop, 1))
            specific_column2 = city_coordinates[:, 0]
            second_row2 = np.tile(specific_column2, (pop, 1))
            X_reshaped = X.reshape(1, -1)
            states = np.array([first_row, second_row, second_row2])
            done = False
            iter = 0
            oldreward = 0
            reward = 0

            probs_history = []
            # 循环前计算初始值
            fitness = np.min(np.array([fun1(pos) for pos in states[0]]))
            # 新设计：在进行强化学习算法之前，先跑一下四个自行算法，获得相应的最优解，强化学习需要比这里的解跑的好，跑的快。
            time1, X1, Best_Pos1, Best_fitness1, IterCurve1 = PSO(pop, dim, ub, lb, fun1, vmax, vmin, EmaxIter,
                                                                  X.copy())  # 877
            print(" 1 ", end="")
            print(Best_fitness1, end="")
            time2, X2, Best_Pos2, Best_fitness2, IterCurve2 = GWO(pop, dim, ub, lb, fun1, EmaxIter, X.copy())  # 607
            print(" 2 ", end="")
            print(Best_fitness2, end="")
            time3, X3, Best_Pos3, Best_fitness3, IterCurve3 = SCA(pop, dim, ub, lb, fun1, EmaxIter, X.copy())  # 960
            print(" 3 ", end="")
            print(Best_fitness3, end="")
            time4, X4, Best_Pos4, Best_fitness4, IterCurve4 = GWO(pop, dim, ub, lb, fun1, EmaxIter, X.copy())  # 607
            print(" 4 ", end="")
            print(Best_fitness4, end="")

            Best_fitness = fitness
            while not done:
                iter = iter + 1
                # print(state.shape)
                action, probs = agent.take_action(states)
                probs_history.append(probs)
                oldreward = reward
                # print(action)
                # states = np.argsort(states)
                time, next_state, reward, done, Best_fitness, Best_Pos, _ = step(action, pop, dim, ub, lb, fun1, vmax,
                                                                                 vmin,
                                                                                 maxIter, state, states, Best_fitness)

                # print(iter)
                transition_dict['statess'].append(states.copy())
                transition_dict['actions'].append(action)
                transition_dict['cost'].append(fitness - Best_fitness)
                transition_dict['next_states'].append(next_state.copy())
                Time_Score = (4-compare_numbers(Best_fitness1, Best_fitness2, Best_fitness3, Best_fitness4,
                                                Best_fitness) )*0
                transition_dict['rewards'].append(reward*5+Time_Score)
                transition_dict['dones'].append(done)
                transition_dict['Best_fitnesss'].append(Best_fitness)

                states = next_state
                episode_return = episode_return + reward  # 总奖励
                if iter >= (EmaxIter / maxIter):
                    done = True
                return_list.append(episode_return)
            # 记录每个episode的累计奖励和最佳适应度
            writer.add_scalar('Reward', reward, i)
            writer.add_scalar('Best Fitness', Best_fitness, i)
            print(" MY ", end="")
            print(Best_fitness, end="")
            # 循环结束后计算全局奖励
            Main_Score=compare_numbers(Best_fitness1,Best_fitness2,Best_fitness3,Best_fitness4,Best_fitness)*compare_numbers(Best_fitness1,Best_fitness2,Best_fitness3,Best_fitness4,Best_fitness)*100
            if Main_Score==1600:
                Main_Score=Main_Score+calculate_difference(Best_fitness1,Best_fitness2,Best_fitness3,Best_fitness4,1, Best_fitness)*1
            else:
                 Main_Score=Main_Score+calculate_difference(Best_fitness1,Best_fitness2,Best_fitness3,Best_fitness4,
                                                            3, Best_fitness)*1




                #定义mainscore如果超过四个算法获得4000分，超过多少值就再给值的10倍分数

            global_reward = Main_Score
            print("G:_ ", end="")
            print(global_reward, end="")

            # writer.add_scalar('Prob', probs, i)
            agent.update(transition_dict, pop, dim,global_reward)
            # if (i_episode+1) % 10 == 0:
            # pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': #'%.3f' %
            # np.mean(return_list[-10:])})
            # pbar.update(1)
            # pbar.set_postfix(Best_fitnesss='{:.2f}'.format(Best_fitness))
            pbar.set_postfix(reward='{:2f}'.format(episode_return))
            pbar.update(1)
            # 更新学习率
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()
    writer.close()
    '''
            # 将 probs_history 转换为 NumPy 数组以便于处理
           
probs_history = np.array(probs_history)

# 绘制每个动作概率的变化
plt.figure(figsize=(10, 6))
for i in range(probs_history.shape[1]):
    plt.plot(probs_history[:, i], label=f"Action {i}")

plt.xlabel("Iterations")
plt.ylabel("Probability")
plt.title("Action Probabilities Over Time")
plt.legend()
plt.show()
            '''

    return return_list, transition_dict, states, Best_fitness, Best_Pos
