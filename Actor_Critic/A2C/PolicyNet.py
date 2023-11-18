#import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import rl_utils

#定义策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, city_num, particle_num, city_feature_num, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        input_dim = city_num * particle_num * city_feature_num
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)  # 使用dim=1来确保softmax沿着正确的维度进行
        return x



