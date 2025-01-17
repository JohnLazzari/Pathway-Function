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
from reward_vis import activity_vis
import json
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity

class Policy(nn.Module):
    def __init__(self, config, hidden_dim, output_dim, device="cuda", noise_act=0.15, noise_inp=0.01):
        super().__init__()

        self.config = config
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = 1

        #parameters for visualizing network activity
        self.activity = np.array([])
        self.count = 0
        
        self.mrnn = mRNN(config, noise_level_act=noise_act, noise_level_inp=noise_inp, device=device)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs, noise=True):
        # Forward pass through mRNN
        x, h = self.mrnn(h, obs[:, None, :], noise=noise)
        # Squeeze in the time dimension (doing timesteps one by one)
        h = h.squeeze(1)
        # Get cortex activity
        m1_act = get_region_activity(self.mrnn, h, "alm")
        # Motor output
        u = self.sigmoid(self.fc(m1_act)).squeeze(dim=1)
        return u, h