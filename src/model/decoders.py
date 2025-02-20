
import torch
import torch.nn as nn
from .attention import NeighborhoodAttention
from .conv_blocks import ConvDownBlock, ConvUpBlock

class WeatherMeshDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_channels_2d,
        output_channels_3d,
        n_pressure_levels,
        n_conv_blocks=3,
        hidden_dim=256
    ):
        super().__init__()
        
        # Transformer layers for initial decoding
        self.transformer_layers = nn.ModuleList([
            NeighborhoodAttention(
                dim=latent_dim,
                depth_window=5,
                height_window=7,
                width_window=7
            ) for _ in range(3)
        ])
        
        # Split into pressure levels and surface paths
        self.split = nn.Conv3d(latent_dim, hidden_dim * (2 ** n_conv_blocks), kernel_size=1)
        
        # Pressure levels (3D) path
        self.pressure_path = nn.ModuleList([
            ConvUpBlock(
                hidden_dim * (2 ** (i + 1)),
                hidden_dim * (2 ** i) if i > 0 else output_channels_3d,
                is_3d=True
            ) for i in reversed(range(n_conv_blocks))
        ])
        
        # Surface (2D) path
        self.surface_path = nn.ModuleList([
            ConvUpBlock(
                hidden_dim * (2 ** (i + 1)),
                hidden_dim * (2 ** i) if i > 0 else output_channels_2d
            ) for i in reversed(range(n_conv_blocks))
        ])
        
    def forward(self, latent):
        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)
            
        # Split features
        features = self.split(latent)
        pressure_features = features[:, :, :-1]
        surface_features = features[:, :, -1:]
        
        # Decode pressure levels
        for block in self.pressure_path:
            pressure_features = block(pressure_features)
            
        # Decode surface features
        surface_features = surface_features.squeeze(2)
        for block in self.surface_path:
            surface_features = block(surface_features)
            
        return surface_features, pressure_features

