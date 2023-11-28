import time
from math import inf

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm, _tqdm

from Actor_Critic.env.step import step
from utils.TSPGenerate import generate_tsp_coordinates
#from test.test1 import YBest_Pos
#from test.test2ac import train
from utils.pltdraw import plot_city_coordinates_line


def train_on_policy_agent(EmaxIter, pop, dim, ub, lb, fun1, vmax, vmin, maxIter, X, agent, num_episodes,
                          city_coordinates, epsilon):
    global current_time
    writer = SummaryWriter('../runs')

    # print("open!!!")
    return_list = []
    minfitness = inf
    with tqdm(total=num_episodes) as pbar:
        for i in range(num_episodes):
            # ... do some work ...
            # print("train" + " " + str(i_episode))
            episode_return = 0
           # city_coordinates = generate_tsp_coordinates(dim, 0, 100)
            transition_dict = {'statess': [], 'actions': [], 'next_states': [], 'rewards': [],'cost': [], 'dones': [],
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
            fitness = np.min(np.array([fun1(pos) for pos in states[0]]))
            Best_fitness =fitness
            while not done:
                iter = iter + 1
                # print(state.shape)
                action,probs = agent.take_action(states, epsilon)
                probs_history.append(probs)
                oldreward = reward
                #print(action)
                #states = np.argsort(states)
                next_state, reward, done, Best_fitness, Best_Pos, _ = step(action, pop, dim, ub, lb, fun1, vmax, vmin,
                                                                           maxIter, state, states,Best_fitness)

                # print(iter)
                transition_dict['statess'].append(states.copy())
                transition_dict['actions'].append(action)
                transition_dict['cost'].append(fitness-Best_fitness)
                transition_dict['next_states'].append(next_state.copy())
                if iter==1:
                    transition_dict['rewards'].append(0)
                else:
                    transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['Best_fitnesss'].append(Best_fitness)

                states = next_state
                episode_return =episode_return+ reward  #总奖励
                if iter >= (EmaxIter / maxIter):
                    done = True
                return_list.append(episode_return)
            # 记录每个episode的累计奖励和最佳适应度
            writer.add_scalar('Reward', reward, i)
            writer.add_scalar('Best Fitness', Best_fitness, i)
            #writer.add_scalar('Prob', probs, i)
            agent.update(transition_dict, pop, dim)
            # if (i_episode+1) % 10 == 0:
            # pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': #'%.3f' %
            # np.mean(return_list[-10:])})
            # pbar.update(1)
            #pbar.set_postfix(Best_fitnesss='{:.2f}'.format(Best_fitness))
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
