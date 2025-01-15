import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import motornet as mn
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity, get_initial_condition
from models import Policy

def main():

    effector = mn.effector.ReluPointMass24()
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)

    device = torch.device("cpu")

    # Parameters for testing
    batch_size = 100
    config = "configurations/mRNN_hyperdirect.json"
    model_save_patorch = "checkpoints/mRNN_reaching.pth"

    # Loading in model
    policy = Policy(config, 50, env.n_muscles, device=device)
    checkpoint = torch.load(model_save_patorch, map_location = torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    def l1(x, y):
        """L1 loss"""
        return torch.mean(torch.sum(torch.abs(x - y), dim=-1))

    # initialize batch
    h = torch.zeros(size=(batch_size, policy.mrnn.total_num_units))
    h = get_initial_condition(policy.mrnn, h)

    obs, info = env.reset(options={"batch_size": batch_size})
    terminated = False

    # initial positions and targets
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        with torch.no_grad():
            action, h = policy(h, obs)
            obs, reward, terminated, truncated, info = env.step(action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = torch.cat(xy, axis=1)
    tg = torch.cat(tg, axis=1)
    loss = l1(xy, tg)  # L1 loss on position

    print(xy.shape)
    print(tg.shape)

if __name__ == "__main__":
    main()