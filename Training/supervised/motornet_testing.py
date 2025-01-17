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

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)

    device = torch.device("cpu")

    # Parameters for testing
    batch_size = 25
    rand_state = False
    config = "configurations/mRNN_thal_inp.json"
    model_save_patorch = "checkpoints/mRNN_thal_inp.pth"

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
    joint_state = torch.zeros(size=(batch_size, 4))

    if rand_state:
        options = {"batch_size": batch_size}
    else:
        options = {"batch_size": batch_size, "joint_state": joint_state}

    obs, info = env.reset(options=options)
    terminated = False

    # initial positions and targets
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        with torch.no_grad():
            action, h = policy(h, obs, noise=False)
            obs, reward, terminated, truncated, info = env.step(action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = torch.cat(xy, axis=1)
    tg = torch.cat(tg, axis=1)
    loss = l1(xy, tg)  # L1 loss on position

    print(f"Testing loss: {loss}")

    colors = plt.cm.inferno(np.linspace(0, 1, len(xy))) 

    for i, batch in enumerate(xy):
        color = colors[i]
        plt.plot(batch[:, 0], batch[:, 1], color=color)
        plt.scatter(batch[0, 0], batch[0, 1], s=150, marker='x', color=color)
        plt.scatter(tg[i, -1, 0], tg[i, -1, 1], s=150, marker='^', color=color)
    plt.show()

if __name__ == "__main__":
    main()