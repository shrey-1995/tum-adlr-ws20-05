"""
Source: https://github.com/cyoon1729/Policy-Gradient-Methods/blob/master/sac/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=3e-3, shared_layer=None):
        super(ValueNetwork, self).__init__()
        if shared_layer is None:
            self.fc1 = nn.Linear(input_dim, 256)
        else:
            self.fc1 = shared_layer
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, shared_layer=None):
        super(SoftQNetwork, self).__init__()
        if shared_layer is None:
            self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        else:
            self.linear1 = shared_layer

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, action_range, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2, shared_layer=None):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if shared_layer is None:
            self.linear1 = nn.Linear(num_inputs, hidden_size)
        else:
            self.linear1 = shared_layer

        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = torch.tensor(action_range)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
               (self.action_range[1] + self.action_range[0]) / 2.0

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        action = self.rescale_action(action)

        log_pi = normal.log_prob(z)
        pi = torch.exp(log_pi.sum(1, keepdim=True))

        return z, action, pi

    def get_probability(self, state, z):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        prob = normal.log_prob(z)

        log_pi = normal.log_prob(z)
        pi = torch.exp(log_pi.sum(1, keepdim=True))
        return pi

