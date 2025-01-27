import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import motornet as mn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity, get_initial_condition
from models import Policy
import config_supervised

def l1(x, y):
    """L1 loss"""
    return torch.mean(torch.sum(torch.abs(x - y), dim=-1))

def main():

    ### PARAMETERS ###
    parser = config_supervised.config_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    if args.arm == "relu":
        effector = mn.effector.ReluPointMass24()
    elif args.arm == "rigid_tendon":
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    else:
        raise ValueError()
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)

    policy = Policy(
        args.model_config_path, 
        50, 
        env.n_muscles, 
        activation_name=args.activation_name,
        noise_level_act=args.noise_level_act, 
        noise_level_inp=args.noise_level_inp, 
        constrained=args.constrained, 
        dt=args.dt,
        t_const=args.t_const,
        device=device
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    losses = []
    interval = 250

    for batch in range(args.epochs):
        # initialize batch
        h = torch.zeros(size=(args.batch_size, policy.mrnn.total_num_units))
        h = get_initial_condition(policy.mrnn, h)
        # Get first timestep
        obs, info = env.reset(options={"batch_size": args.batch_size})
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
        xy = torch.cat(xy, axis=0)
        tg = torch.cat(tg, axis=0)
        loss = l1(xy, tg)  # L1 loss on position
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()
        losses.append(loss.item())
        if (batch % interval == 0) and (batch != 0):
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, args.epochs, sum(losses[-interval:])/interval))

        if batch % args.save_iter == 0:
            torch.save({
                'agent_state_dict': policy.state_dict(),
            }, args.model_save_path)

if __name__ == "__main__":
    main()