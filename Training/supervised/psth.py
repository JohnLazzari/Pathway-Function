from sklearn.cross_decomposition import CCA
import numpy as np
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import numpy as np
import torch
import matplotlib.pyplot as plt
import motornet as mn
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity, get_initial_condition
from models import Policy
import sys
import math

def find_angle(x, y):
    """Finds the angle between the origin and the point (x, y)."""
    angle_deg = math.degrees(math.atan2(y, x))
    if angle_deg < 0:
            angle_deg += 360
    return angle_deg

def get_cond_num(angle, num_conds=8):
    angle_bins = np.linspace(0, 360, num_conds+1)
    for i in range(num_conds):
        if angle > angle_bins[i] and angle < angle_bins[i+1]:
            return i 

def main():

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)
    device = torch.device("cpu")

    # Parameters for testing
    batch_size = 500
    num_conds = 8
    config = "configurations/mRNN_thal_inp.json"
    model_save_patorch = "checkpoints/mRNN_thal_inp.pth"

    # Loading in model
    policy = Policy(config, 50, env.n_muscles, device=device)
    checkpoint = torch.load(model_save_patorch, map_location = torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    # initialize batch
    h = torch.zeros(size=(batch_size, policy.mrnn.total_num_units))
    h = get_initial_condition(policy.mrnn, h)
    joint_state = torch.zeros(size=(batch_size, 4))

    obs, info = env.reset(options={"batch_size": batch_size, "joint_state": joint_state})
    terminated = False

    # initial positions and targets
    xy = [info["states"]["fingertip"][:, None, :]]
    tg = [info["goal"][:, None, :]]
    all_hs = [h[:, None, :]]

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
        with torch.no_grad():
            action, h = policy(h, obs, noise=False)
            obs, reward, terminated, truncated, info = env.step(action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            all_hs.append(h[:, None, :])

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = torch.cat(xy, axis=1)
    tg = torch.cat(tg, axis=1)
    all_hs = torch.cat(all_hs, axis=1)

    colors = plt.cm.inferno(np.linspace(0, 1, num_conds)) 

    # Need to create conditions based on reach angle
    act_conds = {}
    
    # Add activity for a specific condition into condition specific dictionary
    for i, batch in enumerate(tg):
        tg_angle = find_angle(batch[-1, 0], batch[-1, 1])
        tg_cond = get_cond_num(tg_angle, num_conds=num_conds)
        if tg_cond not in act_conds:
            act_conds[tg_cond] = []
        act_conds[tg_cond].append(all_hs[i])
    
    for cond in range(num_conds):
        act_conds[cond] = torch.stack(act_conds[cond], dim=0)

    fig = plt.figure()
    for i, region in enumerate(policy.mrnn.region_dict):
        ax = fig.add_subplot(2, 4, i+1)
        for cond in range(num_conds):
            all_hs_region = get_region_activity(policy.mrnn, act_conds[cond], region)
            all_hs_reduced = np.mean(all_hs_region.numpy(), axis=-1)
            all_hs_reduced = np.reshape(all_hs_reduced, [all_hs_region.shape[0], all_hs_region.shape[1]])
            trial_averaged_act = np.mean(all_hs_reduced, axis=0)
            ax.plot(trial_averaged_act, color=colors[cond], label=f"Cond {cond}")
            ax.set_title(region)
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    main()