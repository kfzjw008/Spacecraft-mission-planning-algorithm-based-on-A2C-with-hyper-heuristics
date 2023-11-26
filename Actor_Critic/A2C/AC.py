import random

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from Actor_Critic.A2C.PolicyNet import PolicyNet
from Actor_Critic.A2C.ValueNet import ValueNet
#from test.test2ac import action_dims


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        self.value_losses = []  # 用于存储价值损失
        self.td_errors = []  # 用于存储TD误差
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=0.9)
        self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=0.9)
        self.gamma = gamma
        self.device = device

    def take_action(self, states, epsilon):
        states = torch.tensor([states], dtype=torch.float).to(self.device)
        probs = self.actor(states)
        if random.random() < epsilon:
            # 随机探索
            action = np.random.choice(4)  # 假设 self.num_actions 是动作的数量
        else:
            # 利用

            probs = probs.cpu().detach().numpy().squeeze()  # 转移到 CPU，转换为 NumPy 数组，压缩维度
            formatted_probs = [f"{prob:.4f}" for prob in probs]  # 格式化输出，保留小数点后四位
            #print(formatted_probs)
            #print(probs)
            action_dist = torch.distributions.Categorical(torch.tensor(probs))
            action = action_dist.sample().item()
        return action,probs

    def update(self, transition_dict, pop, dim):
        writer = SummaryWriter('../runs')
        statess = torch.tensor(transition_dict['statess'],
                               dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        #print(self.critic(statess))
        td_delta = td_target - self.critic(statess)
        #td_target.cpu().detach().numpy()
        # self.critic(statess).cpu().detach().numpy()
        # 时序差分误差
        log_probs = torch.log(self.actor(statess).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean( F.mse_loss(self.critic(statess), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        # 计算价值损失和TD误差
        #critic_loss = torch.mean(F.mse_loss(self.critic(statess), td_target.detach()))
        td_error = (td_target - self.critic(statess)).detach().cpu().numpy()

        # 将损失和误差添加到列表中
        self.value_losses.append(critic_loss.item())

        self.td_errors.append(np.mean(np.abs(td_error)))
        print("loss=" + str(critic_loss.item()) + " td:" )
        writer.add_scalar('loss', int(critic_loss.item()))
        writer.close()

        if 1==0:
            # 绘制价值损失曲线
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.value_losses)
            plt.title("Value Loss Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("Value Loss")

            # 绘制TD误差曲线
            plt.subplot(1, 2, 2)
            plt.plot(self.td_errors)
            plt.title("TD Error Over Time")
            plt.xlabel("Iteration")
            plt.ylabel("TD Error")

            plt.show()

        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
