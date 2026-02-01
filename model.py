# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils.utils import RevIN1d

class PatchEncoder(nn.Module): #Simple 1D CNN with RevIN
    def __init__(self, in_channels=1, projection_dim=256, layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin: bool = True,       
                 revin_affine: bool = False,   
                 revin_eps: float = 1e-5,      
                 revin_min_sigma: float = 1e-5 
                 ):
        super(PatchEncoder, self).__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # RevIN 
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        #  Conv blocks 
        self.convblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(layers[i - 1] if i > 0 else in_channels, self.layers[i],
                          kernel_size=self.kss[i], stride=1, padding=self.kss[i] // 2, bias=False),
                nn.BatchNorm1d(self.layers[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(self.layers))
        ])

        # Heads 
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        self.classification_head = nn.Linear(self.layers[-1]*2, 1)

    def forward(self, x, return_embedding=False, return_projection=False):
       
        if self.revin is not None:
            x = self.revin.norm(x)  

        for block in self.convblocks:
            x = block(x)

        h = self.fc_embedding(x).flatten(start_dim=1)  # (N, D)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        return self.projection_head(h)

