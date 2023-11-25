#import gym
import numpy as np
import matplotlib.pyplot as plt
from utils import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
#定义策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # Assuming input_dim = (dim, pop, distance)
        self.conv1 = nn.Conv2d(in_channels=input_dim[2], out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=16 * 5 * 48,
                             out_features=hidden_dim)  # Adjust the in_features according to the output of conv2
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=action_dim)
        # 初始化权重和偏置
        #self.init_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 48)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_scores = self.fc3(x)
        return F.softmax(action_scores, dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 初始化权重为0
                nn.init.constant_(m.weight, 0)
                # 初始化偏置为0
                nn.init.constant_(m.bias, 0)




