import torch
import matplotlib.pyplot as plt
import time
import random

from Actor_Critic.A2C.PolicyNet import PolicyNet
from Actor_Critic.A2C.ValueNet import ValueNet
from Actor_Critic.env.step import step
from Actor_Critic.train import train_on_policy_agent
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

#配置项
#random.seed(3)
#np.random.seed(4)
#torch.manual_seed(0)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
timess=0

#参数设置
train=0#0为重新训练模式，1为加载后训练，2为测试模式
pop = 200 #粒子总数
dim = 30 #数据维度，状态空间维度
ub = np.ones(dim) * 1 #粒子上界
lb = np.ones(dim) * -1#粒子下界
vmax = (ub - lb) * 0.1 #粒子速度最大值，PSO专用
vmin = -vmax #粒子速度最小值，PSO专用
maxIter = 5 #算法一次最大迭代次数
EmaxIter =50 #一轮次内最大的算法迭代次数
cmin=0 #初始坐标最小值
cmax=100 #初始坐标最大值
action_dims =4 #动作空间维度
actor_lr = 1e-3 # 学习速率，用于更新Actor模型的参数
critic_lr = 1e-2 # 学习速率，用于更新Critic模型的参数
num_episodes = 50 # 总的训练轮次（即训练多少个episodes）
hidden_dim = 120 # 隐藏层的维度，影响模型的复杂度和表达能力
gamma = 0.98 # 折扣因子，用于计算奖励的折现值，影响智能体对未来奖励的考虑程度
policy_net_path = '../Actor_Critic/Net/policy_net.pth'
value_net_path = '../Actor_Critic/Net/value_net.pth'
from Actor_Critic.A2C.AC import ActorCritic
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using GPU!")
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU.")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device( "cpu")

#env_name = 'CartPole-v0'
#env = gym.make(env_name)
#env.seed(0)

#函数设置
#city_coordinates = generate_tsp_coordinates(dim, cmin, cmax)
city_coordinates =[(23.796462709189136, 54.42292252959518), (36.99551665480792, 60.39200385961945), (62.572030410805404, 6.552885923981311), (1.3167991554874137, 83.746908209646), (25.935401432800763, 23.433096104669637), (99.56448355104628, 47.026350752244795), (83.64614512743887, 47.635320869933494), (63.906814054416195, 15.061642402352394), (63.486065828518846, 86.80453071432967), (52.31812103833013, 74.12518562014903), (67.14114753695925, 6.403143822699731), (75.82302462868174, 59.10995829313176), (30.126765951571233, 3.1011751469749993), (86.55272369789456, 47.27490886654668), (71.88239240658031, 87.88128002554816), (71.41294836112026, 92.10986675838745), (39.496340400074395, 80.09087709852282), (44.46210560507606, 93.55867217045211), (87.88666603380416, 9.745430973087721), (13.59688602006689, 21.698694123313732), (96.5480138898203, 43.616186662742926), (62.6648290866804, 30.10261984255054), (50.72429838290595, 38.58662588449025), (35.091048877018004, 58.50741074053635), (58.425179297019895, 90.4201770847775), (68.19821366349666, 92.8945601200017), (85.64005663967556, 99.09896448688151), (67.12735421625182, 16.309962197106977), (86.06375331162683, 96.46329473090614), (90.46959845122366, 56.91075034743235)]

def main():
    start_time = time.time()
    state_dim = (dim, pop)
    action_dim = action_dims
    # 生成初始解
    X = initialization(pop, ub, lb, dim)
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
        Best_Pos=X
        # 将X从NumPy数组转换为PyTorch张量

        for _ in range(200):
            X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
            with torch.no_grad():
                # 通过模型获取动作
                outputs = loaded_policy_net(X_tensor)
                action_probabilities = outputs.data  # 获取网络输出的数据

                action = torch.argmax(action_probabilities, dim=1) # 选择概率最大的动作
                print(action)
        next_state, reward, done, Best_fitness, Best_Pos, _ = step(action.item(), pop, dim, ub, lb, fun1, vmax, vmin,
                                                                   maxIter, X)
        transition_dict['states'].append(X)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        transition_dict['Best_fitnesss'].append(Best_fitness)
        X = next_state
    end_time = time.time()
    elapsed_time = end_time - start_time



    if train == 0 or train == 1:
        # 算法调用


        agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
    if  train == 1:
        agent.actor = loaded_policy_net
        agent.critic = loaded_value_net

    if train == 0 or train == 1:
        start_time = time.time()
        return_list, transition_dict, state, Best_fitness, Best_Pos = train_on_policy_agent(EmaxIter, pop, dim, ub, lb,fun1, vmax, vmin, maxIter,X, agent, num_episodes)

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

    #action_counts.action_counts(transition_dict, action_counts)


    # 画图
    if train == 0 or train == 1:
        plta2c(return_list)
    # plot_iterations(IterCurve)
    plot_city_coordinates(city_coordinates)
    plot_city_coordinates_line(city_coordinates, YBest_Pos)


def fun1(x):
    global timess  # 声明要使用的全局变量
    global city_coordinates
    # 假设X是一个numpy数组
    # 对X进行排序，并返回对应的索引
    start_time = time.time()
    Y = np.argsort(x)
    total_distance = calculate_total_distance(Y, city_coordinates)
    end_time = time.time()

    elapsed_time = end_time - start_time
    timess=timess+elapsed_time
    return total_distance
   # return x[0]**2 + x[1]**2

if __name__ == "__main__":
    main()