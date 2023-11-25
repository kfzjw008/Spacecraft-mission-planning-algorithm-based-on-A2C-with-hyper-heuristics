#import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

from utils import rl_utils
class ValueNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # Assuming input_dim = (dim, pop, distance)
        self.conv1 = nn.Conv2d(in_channels=input_dim[2], out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=16 * 5 * 48,
                             out_features=120)  # Adjust the in_features according to the output of conv2
        self.fc2 = nn.Linear(in_features=120, out_features=hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 48)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value
