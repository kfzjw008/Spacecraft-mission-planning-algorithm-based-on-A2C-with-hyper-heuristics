#import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import rl_utils
class ValueNet(torch.nn.Module):
    def __init__(self, city_num, particle_num, city_feature_num, hidden_dim):
        super(ValueNet, self).__init__()
        input_dim = city_num * particle_num * city_feature_num
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
