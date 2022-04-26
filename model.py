from math import perm
import torch 
from torch import nn
import random 
import numpy as np 

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class DeepNormal(nn.Module):
    #https://romainstrock.com/blog/modeling-uncertainty-with-pytorch.html
    def __init__(self, n_inputs, n_hidden):
        super().__init__()

        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )
        
        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
        )
        
        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
            nn.Softplus(),  # enforces positivity
        )
             
    def forward(self, x):
        # Shared embedding
        shared = self.shared_layer(x)
        
        # Parametrization of the mean
        μ = self.mean_layer(shared)
        
        # Parametrization of the standard deviation
        σ = self.std_layer(shared)
        
        return torch.distributions.Normal(μ, σ)
  
def deep_normal_compute_loss(normal_dist, y):
    neg_log_likelihood = -normal_dist.log_prob(y)
    return torch.mean(neg_log_likelihood)

class LearnHopperPenalty:
    def __init__(self, seed, state_dim=15, action_dim=3):
        set_random_seed(seed)
        input_size = state_dim

        self.model = DeepNormal(n_inputs=state_dim+action_dim, n_hidden=48)

        self.optim = torch.optim.Adam(self.model.parameters())
        self.loss = deep_normal_compute_loss
        
    def step(self, state, action):
        return self.model(torch.cat((torch.from_numpy(state), torch.from_numpy(action))).float())

    def train(self, step_info):
        #Just doing SGD for now
        permutation_idxes = np.random.permutation(len(step_info))
        permuted_step_info = [step_info[i] for i in permutation_idxes]
        states_before = [step[0] for step in permuted_step_info]
        actions = [step[1] for step in permuted_step_info]
        electricity_costs = [step[2] for step in permuted_step_info]
        for i in range(len(permuted_step_info)):
            state_before = states_before[i]
            action = actions[i]
            electricity_cost = electricity_costs[i]
            self.optim.zero_grad()
            loss = self.loss(self.model(torch.cat((torch.from_numpy(state_before), torch.from_numpy(action))).float()), torch.tensor(electricity_cost))
            loss.backward()
            self.optim.step()