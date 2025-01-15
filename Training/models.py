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
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.to(device)

    def forward(self, h, obs):
        x, h = self.mrnn(h, obs[:, None, :])
        h = h.squeeze(1)
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