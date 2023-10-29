#import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import rl_utils

#定义策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        a=state_dim[0] * state_dim[1]
        #self.fc1 = torch.nn.Linear(state_dim[0] * state_dim[1], hidden_dim)
        self.fc1 = torch.nn.Linear(a, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        #print(x.shape)
        batch_size, _, _ = x.size()  # 获取批处理大小
        x = x.view(batch_size, -1)  # 将 x 重塑为 [batch_size, num_features]
        #print(x.shape)
        x=self.fc1(x)
        x = F.relu(x)
        return F.softmax(self.fc2(x), dim=0)


