import os
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity, get_initial_condition

effector = mn.effector.ReluPointMass24()
env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)

class Policy(th.nn.Module):
    def __init__(self, config, hidden_dim, output_dim, device):
        super().__init__()

        self.config = config
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = 1

        #parameters for visualizing network activity
        self.activity = np.array([])
        self.count = 0
        
        self.mrnn = mRNN(config, device=device)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs):
        x, h = self.mrnn(h, obs[None, :, :])
        h = h.squeeze(0)
        m1_act = get_region_activity(self.mrnn, h, "alm")
        u = self.sigmoid(self.fc(m1_act)).squeeze(dim=1)
        u_ = u.clone().norm()
        self.activity = np.append(self.activity, u_.detach().numpy())
        self.count += 1
        if self.count == 5000:
            print(self.activity)
            plt.plot(self.activity)
            plt.show()
        return u, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    
device = th.device("cpu")

config = "configurations/mRNN_hyperdirect.json"
model_save_path = "checkpoints/mRNN_reaching"
policy = Policy(config, 50, env.n_muscles, device=device)
optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)

batch_size = 1
n_batch = 10000
losses = []
interval = 250

def l1(x, y):
    """L1 loss"""
    return th.mean(th.sum(th.abs(x - y), dim=-1))

for batch in range(n_batch):
    # initialize batch
    h = th.zeros(size=(batch_size, policy.mrnn.total_num_units))
    h = get_initial_condition(policy.mrnn, h)
    obs, info = env.reset(options={"batch_size": batch_size})
    terminated = False

    # initial positions and targets
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        action, h = policy(h, obs)
        obs, reward, terminated, truncated, info = env.step(action=action)

        xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
        tg.append(info["goal"][:, None, :])  # targets

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = th.cat(xy, axis=0)
    tg = th.cat(tg, axis=0)
    loss = l1(xy, tg)  # L1 loss on position
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()
    losses.append(loss.item())
    if (batch % interval == 0) and (batch != 0):
        print("Batch {}/{} Done, mean policy loss: {}".format(batch, n_batch, sum(losses[-interval:])/interval))

    if batch % 1000 == 0:
        th.save({
            'agent_state_dict': policy.state_dict(),
        }, model_save_path + '.pth')