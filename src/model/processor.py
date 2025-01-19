import torch.nn as nn
from .attention import NeighborhoodAttention

class WeatherMeshProcessor(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_layers=10,
        depth_window=5,
        height_window=7,
        width_window=7
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            NeighborhoodAttention(
                dim=latent_dim,
                depth_window=depth_window,
                height_window=height_window,
                width_window=width_window
            ) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

