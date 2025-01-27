import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import motornet as mn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity, get_initial_condition, manipulation_stim
from models import Policy

def main():

    # Parameters for testing
    batch_size = 5
    rand_state = True
    arm = "relu"            # relu or rigid tendon
    config = "Training/configurations/mRNN_thal_inp.json"
    model_save_patorch = "Training/checkpoints/mRNN_thal_inp_relu_2.pth"
    device = torch.device("cpu")
    perturbation = False
    start_silence = 0
    end_silence = 100
    region_perturbed_list = ["d2"]
    stim_strength = 10

    if arm == "rigid_tendon":
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    elif arm == "relu":
        effector = mn.effector.ReluPointMass24()
    else:
        raise ValueError("Only two arms implemented")

    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)

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
    joint_state = torch.tensor([(effector.pos_lower_bound[0] + effector.pos_upper_bound[0]) / 2, (effector.pos_lower_bound[1] + effector.pos_upper_bound[1]) / 2, 0, 0]).unsqueeze(0).repeat(batch_size, 1)

    if rand_state:
        options = {"batch_size": batch_size}
    else:
        options = {"batch_size": batch_size, "joint_state": joint_state}

    obs, info = env.reset(options=options)
    terminated = False

    # initial positions and targets
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]

    timesteps = 0
    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        with torch.no_grad():
            if perturbation == True and timesteps > start_silence and timesteps < end_silence:
                mask = torch.zeros(size=(1, 1, policy.mrnn.total_num_units), device="cpu")
                for region in region_perturbed_list:
                    cur_mask = stim_strength * (policy.mrnn.region_mask_dict[region])
                    mask = mask + cur_mask
                stim = mask
            else:
                stim = torch.zeros(size=(1, 1, policy.mrnn.total_num_units))

            action, h = policy(h, obs, stim, noise=False)
            obs, reward, terminated, truncated, info = env.step(action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            timesteps += 1

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