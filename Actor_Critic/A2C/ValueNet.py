#import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import rl_utils
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()

        a = state_dim[0] * state_dim[1]
        # self.fc1 = torch.nn.Linear(state_dim[0] * state_dim[1], hidden_dim)
        self.fc1 = torch.nn.Linear(a, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x,pop,dim):
        #print(x.shape)
        x = x.view(-1, pop*dim)  # 调整形状，-1 会自动计算批的大小
        #print(x.shape)
        x=self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)