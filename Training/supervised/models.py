import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity

class Policy(nn.Module):
    def __init__(
            self, 
            config, 
            hidden_dim, 
            output_dim, 
            activation_name="relu",
            noise_level_act=0.01, 
            noise_level_inp=0.01, 
            constrained=True, 
            dt=10,
            t_const=100,
            batch_first=True,
            lower_bound_rec=0,
            upper_bound_rec=10,
            lower_bound_inp=0,
            upper_bound_inp=10,
            device="cuda"
        ):
        super().__init__()

        self.config = config
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = 1
        self.constrained = constrained
        self.device = device
        self.dt = dt
        self.t_const = t_const
        self.batch_first = batch_first
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        self.activation_name = activation_name
        self.lower_bound_rec = lower_bound_rec
        self.upper_bound_rec = upper_bound_rec
        self.lower_bound_inp = lower_bound_inp
        self.upper_bound_inp = upper_bound_inp

        #parameters for visualizing network activity
        self.activity = np.array([])
        self.count = 0
        
        self.mrnn = mRNN(
            config=config, 
            activation=activation_name,
            noise_level_act=noise_level_act, 
            noise_level_inp=noise_level_inp, 
            constrained=constrained, 
            dt=dt,
            t_const=t_const,
            batch_first=batch_first,
            lower_bound_rec=lower_bound_rec,
            upper_bound_rec=upper_bound_rec,
            lower_bound_inp=lower_bound_inp,
            upper_bound_inp=upper_bound_inp,
            device=device
        )

        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs, noise=True):
        # Forward pass through mRNN
        x, h = self.mrnn(h, obs[:, None, :], noise=noise)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        # Get cortex activity
        m1_act_exc = get_region_activity(self.mrnn, h, "alm_exc")
        # Motor output
        u = self.sigmoid(self.fc(m1_act_exc)).squeeze(dim=1)
        return u, h