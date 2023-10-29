import torch
import matplotlib.pyplot as plt
import time
import random

from Actor_Critic.A2C.PolicyNet import PolicyNet
from Actor_Critic.A2C.ValueNet import ValueNet
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
from utils.pltdraw import plot_city_coordinates, plot_iterations, plot_city_coordinates_line, plta2c

from utils import rl_utils, action_counts

#配置项
random.seed(3)
np.random.seed(4)
torch.manual_seed(0)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
timess=0

#参数设置
pop = 100 #粒子总数
dim = 30 #数据维度，状态空间维度
ub = np.ones(dim) * 1 #粒子上界
lb = np.ones(dim) * -1#粒子下界
vmax = (ub - lb) * 0.1 #粒子速度最大值，PSO专用
vmin = -vmax #粒子速度最小值，PSO专用
maxIter = 2 #算法一次最大迭代次数
EmaxIter =20 #一轮次内最大的算法迭代次数
cmin=0 #初始坐标最小值
cmax=100 #初始坐标最大值
action_dims =4 #动作空间维度
actor_lr = 1e-3 # 学习速率，用于更新Actor模型的参数
critic_lr = 1e-2 # 学习速率，用于更新Critic模型的参数
num_episodes = 20 # 总的训练轮次（即训练多少个episodes）
hidden_dim = 120 # 隐藏层的维度，影响模型的复杂度和表达能力
gamma = 0.98 # 折扣因子，用于计算奖励的折现值，影响智能体对未来奖励的考虑程度
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
def fun1(x):
    global timess  # 声明要使用的全局变量
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


#测试和执行
state_dim = (dim, pop)
action_dim = action_dims
#生成初始解
X=initialization(pop, ub, lb, dim)
#生成初始坐标
city_coordinates = generate_tsp_coordinates(dim,cmin,cmax)


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


#算法调用
start_time = time.time()

agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)
return_list,transition_dict,state,Best_fitness,Best_Pos = train_on_policy_agent(EmaxIter,pop, dim,ub, lb, fun1,vmax,vmin,
    maxIter, X, agent, num_episodes)

end_time = time.time()
elapsed_time = end_time - start_time

# 保存模型
torch.save(agent.actor.state_dict(), '../Actor_Critic/Net/policy_net.pth')
torch.save(agent.critic.state_dict(), '../Actor_Critic/Net/value_net.pth')
#结果输出

YBest_Pos = np.argsort(Best_Pos)
print("最优位置:", Best_Pos)
print("最优路径:", YBest_Pos)
print("最优适应度值:", Best_fitness)
print("用时:",elapsed_time)
print("实际算法用时：",elapsed_time-timess)
print("算法时间占比：",(elapsed_time-timess)/(elapsed_time))
#动作统计

action_counts.action_counts(transition_dict,action_counts)

'''
with torch.no_grad():  # 关闭梯度计算，节省内存，加速计算
    predictions = model(new_data)  # 得到模型的预测结果

'''

#画图
plta2c(return_list)
#plot_iterations(IterCurve)
plot_city_coordinates(city_coordinates)
plot_city_coordinates_line(city_coordinates, YBest_Pos)

