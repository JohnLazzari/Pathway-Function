import os
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity, get_initial_condition
from models import Policy

def main():

    device = th.device("cuda")
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)

    config = "configurations/mRNN_thal_inp.json"
    model_save_path = "checkpoints/mRNN_thal_inp"
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
        # Get first timestep
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

if __name__ == "__main__":
    main()