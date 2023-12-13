import datetime
from math import inf

import torch
import matplotlib.pyplot as plt
import time
import random
from Actor_Critic.A2C.PolicyNet import PolicyNet
from Actor_Critic.A2C.ValueNet import ValueNet
from Actor_Critic.env.step import step
from Actor_Critic.train import train_on_policy_agent
# from Actor_Critic.train import train_on_policy_agent
from Algorithm.GWO import GWO
from Algorithm.PSO import PSO
from Algorithm.SCA import SCA
from Algorithm.TSA import TSA
from fun import fun1
from utils.TSPGenerate import generate_tsp_coordinates
from utils.calculate_total_distance import calculate_total_distance
from utils.initialization import initialization
import numpy as np
import matplotlib.pyplot as plt

from utils.os import delete_model_files
from utils.pltdraw import plot_city_coordinates, plot_iterations, plot_city_coordinates_line, plta2c

from utils import rl_utils, action_counts

# 配置项
# random.seed(3)
# np.random.seed(4)
# torch.manual_seed(0)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
timess = 0

# 参数设置
randomD = 0  # 指定坐标与否，0为指定，1为随机
train = 0  # 0为重新训练模式，1为加载后训练，2为测试模式
pop = 200  # 粒子总数
dim = 30  # 数据维度，状态空间维度
distance = 3  # 参数维度，可以理解为是任务级别指令的参数值+1
ub = np.ones(dim) * 1  # 粒子上界
lb = np.ones(dim) * -1  # 粒子下界
vmax = (ub - lb) * 0.1  # 粒子速度最大值，PSO专用
vmin = -vmax  # 粒子速度最小值，PSO专用
maxIter = 5  # 算法一次最大迭代次数
EmaxIter = 500  # 一轮次内最大的算法迭代次数
cmin = 0  # 初始坐标最小值
cmax = 100  # 初始坐标最大值
action_dims = 4  # 动作空间维度
actor_lr = 2e-4# 学习速率，用于更新Actor模型的参数
critic_lr = 3e-3  # 学习速率，用于更新Critic模型的参数
num_episodes = 10000  # 总的训练轮次（即从头到尾收敛次数，训练多少个episodes）
hidden_dim = 120  # 隐藏层的维度，影响模型的复杂度和表达能力
gamma = 0.95  # 折扣因子，用于计算奖励的折现值，影响智能体对未来奖励的考虑程度
entropy_beta=0.05
# 在训练循环中动态调整ε
epsilon_start = 1.5
epsilon_end = 0.01
epsilon_decay = 500
policy_net_path = '../Actor_Critic/Net/policy_net.pth'
value_net_path = '../Actor_Critic/Net/value_net.pth'
from Actor_Critic.A2C.AC import ActorCritic

# global current_time = time.strftime("%Y%m%d-%H%M%S")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using GPU!")
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU.")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# env_name = 'CartPole-v0'
# env = gym.make(env_name)
# env.seed(0)
# 函数设置
if randomD == 1:
    city_coordinates = generate_tsp_coordinates(dim, cmin, cmax)
if randomD == 0:
    c=1


def main(city_coordinates):

    global nYBest_Pos, minfitness

    start_time = time.time()
    # state_dim = (dim, pop)
    state_dim = (dim, pop, distance)
    action_dim = action_dims
    # 生成初始解
    X = initialization(pop, ub, lb, dim)
    # X = np.argsort(X)
    # 生成初始坐标

    if train == 0:
        delete_model_files(policy_net_path, value_net_path)
    if train == 1 or train == 2:
        loaded_policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
        loaded_value_net = ValueNet(state_dim, hidden_dim)
        loaded_policy_net.load_state_dict(torch.load(policy_net_path))
        loaded_value_net.load_state_dict(torch.load(value_net_path))
        loaded_policy_net.to(device)
        loaded_value_net.to(device)
    if train == 1:
        loaded_policy_net.train()
        loaded_value_net.train()
    if train == 2:
        loaded_policy_net.eval()
        loaded_value_net.eval()

    '''
    # 创建新的模型实例
    loaded_policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
    loaded_value_net = ValueNet(state_dim, hidden_dim)

    # 加载保存的参数
    loaded_policy_net.load_state_dict(torch.load('policy_net.pth'))
    loaded_value_net.load_state_dict(torch.load('value_net.pth'))

    # （可选）设置为评估模式
    loaded_policy_net.eval()
    loaded_value_net.eval()
    '''

    if train == 2:
        start_time = time.time()
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],
                           'Best_fitnesss': []}
        X = initialization(pop, ub, lb, dim)

        Best_Pos = X
        # 将X从NumPy数组转换为PyTorch张量
        first_row = X
        city_coordinates = np.array(city_coordinates)
        specific_column = city_coordinates[:, 1]
        second_row = np.tile(specific_column, (pop, 1))
        specific_column2 = city_coordinates[:, 0]
        second_row2 = np.tile(specific_column2, (pop, 1))
        X_reshaped = X.reshape(1, -1)
        states = np.array([first_row, second_row, second_row2])
        minfitness = inf
        IterCurve = np.zeros(1000)

        for i in range(1000):
            print(i)
            X_tensor = torch.from_numpy(states).float().unsqueeze(0).to(device)
            with torch.no_grad():
                # 通过模型获取动作
                outputs = loaded_policy_net(X_tensor)
                action_probabilities = outputs.data  # 获取网络输出的数据

                # action = torch.argmax(action_probabilities, dim=1)  # 选择概率最大的动作
                m = torch.distributions.Categorical(action_probabilities)  # 创建一个分类分布
                # 输出采样分布的概率
                print("采样分布的概率:", m.probs)
                action = m.sample()  # 从这个分布中采样一个动作
                print(action)
                fitness = np.min(np.array([fun1(pos) for pos in states[0]]))
                Best_fitness = fitness
            timesss, next_state, reward, done, Best_fitness, Best_Pos, _ = step(action.item(), pop, dim, ub, lb, fun1,vmax, vmin,1, X, states, Best_fitness)
            transition_dict['states'].append(X)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['Best_fitnesss'].append(Best_fitness)


            #IterCurve
            states = next_state
            YBest_Pos = np.argsort(Best_Pos)
            # print("最优位置:", Best_Pos)
            # print("最优路径:", YBest_Pos)
            # print("最优适应度值:", Best_fitness)
            if minfitness > Best_fitness:
                mbestpos = Best_Pos
                minfitness = Best_fitness
                nYBest_Pos = YBest_Pos

                print("最优适应度值:", minfitness)
                city_coordinates_array = np.array(city_coordinates)
                plot_city_coordinates_line(city_coordinates_array, nYBest_Pos)

            # print("用时:", elapsed_time)
            # print("实际算法用时：", elapsed_time - timess)
            # print("算法时间占比：", (elapsed_time - timess) / (elapsed_time))
            IterCurve[i] = minfitness
    end_time = time.time()
    elapsed_time = end_time - start_time

    if train == 0 or train == 1:
        # 算法调用

        agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device,entropy_beta,
                            epsilon_start,
                            epsilon_end,epsilon_decay )
    if train == 1:
        agent.actor = loaded_policy_net
        agent.critic = loaded_value_net

    if train == 0 or train == 1:
        start_time = time.time()
        return_list, transition_dict, state, Best_fitness, Best_Pos = train_on_policy_agent(EmaxIter, pop, dim, ub,
                                                                                            lb, fun1, vmax, vmin,
                                                                                            maxIter, X, agent,
                                                                                            num_episodes,
                                                                                            city_coordinates,file)

        end_time = time.time()
        elapsed_time = end_time - start_time

    # 保存模型
    if train == 0 or train == 1:
        torch.save(agent.actor.state_dict(), policy_net_path)
        torch.save(agent.critic.state_dict(), value_net_path)
        # 结果输出

        YBest_Pos = np.argsort(Best_Pos)
        print("最优位置:", Best_Pos)
        print("最优路径:", YBest_Pos)
        print("最优适应度值:", Best_fitness)
        print("用时:", elapsed_time)
        print("实际算法用时：", elapsed_time - timess)
        print("算法时间占比：", (elapsed_time - timess) / (elapsed_time))
    # 动作统计

    # action_counts.action_counts(transition_dict, action_counts)

    # 画图
    if (train == 0 or train == 1) and num_episodes > 10:
        plta2c(return_list)
    # plot_iterations(IterCurve)
    plot_city_coordinates(city_coordinates)
    city_coordinates_array = np.array(city_coordinates)

    if train == 2:
        print("最优位置:", mbestpos)
        print("最优路径:", nYBest_Pos)
        print("最优适应度值:", minfitness)
        print("用时:", elapsed_time)
        print("实际算法用时：", elapsed_time - timess)
        print("算法时间占比：", (elapsed_time - timess) / (elapsed_time))
        plot_iterations(IterCurve)
        plot_city_coordinates_line(city_coordinates_array, nYBest_Pos)
    else:
        plot_city_coordinates_line(city_coordinates_array, YBest_Pos)


def fun1(x):
    global timess  # 声明要使用的全局变量
    global city_coordinates
    # 假设X是一个numpy数组
    # 对X进行排序，并返回对应的索引
    start_time = time.time()
    Y = np.argsort(x)
    total_distance = calculate_total_distance(Y, city_coordinates)
    #print("函数计算", file=file)
    #print(x, file=file)
    #print(Y, file=file)
    #print(total_distance, file=file)
    #print(city_coordinates, file=file)
    #print("函数计算end", file=file)
    end_time = time.time()

    elapsed_time = end_time - start_time
    timess = timess + elapsed_time
    return total_distance


# return x[0]**2 + x[1]**2

if __name__ == "__main__":
    city_coordinates = generate_tsp_coordinates(dim, cmin, cmax)
    print(city_coordinates)
    # 获取当前时间
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'{formatted_time}'
    file_name=file_name+"_"+str(pop)+"_"+str(dim)+"_"+str(
        maxIter)+"_"+str(EmaxIter)+"_"+str(num_episodes)+".txt"

    with open("../log/"+file_name, 'a') as file:
        main(city_coordinates)
