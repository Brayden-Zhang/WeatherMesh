"""
Input: analysis from ERA5/HRES/GFS dataset. 


"""


import torch
import torch.nn as nn
from .attention import NeighborhoodAttention
from .conv_blocks import ConvDownBlock, ConvUpBlock

class WeatherMeshEncoder(nn.Module):
    def __init__(
        self, 
        input_channels_2d: int,
        input_channels_3d: int,
        latent_dim: int,
        n_pressure_levels: int,
        n_conv_blocks: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Surface (2D) path
        self.surface_path = nn.ModuleList([
            ConvDownBlock(
                input_channels_2d if i == 0 else hidden_dim * (2 ** i),
                hidden_dim * (2 ** (i + 1))
            ) for i in range(n_conv_blocks)
        ])
        
        # Pressure levels (3D) path
        self.pressure_path = nn.ModuleList([
            ConvDownBlock(
                input_channels_3d if i == 0 else hidden_dim * (2 ** i),
                hidden_dim * (2 ** (i + 1)),
                is_3d=True
            ) for i in range(n_conv_blocks)
        ])
        
        # Transformer layers for final encoding
        self.transformer_layers = nn.ModuleList([
            NeighborhoodAttention(
                dim=latent_dim,
                depth_window=5,
                height_window=7,
                width_window=7
            ) for _ in range(3)
        ])
        
        # Final projection to latent space
        self.to_latent = nn.Conv3d(
            hidden_dim * (2 ** n_conv_blocks),
            latent_dim,
            kernel_size=1
        )

    def forward(self, x_2d, x_3d):
        # Process surface data
        surface_features = x_2d
        for block in self.surface_path:
            surface_features = block(surface_features)
            
        # Process pressure level data
        pressure_features = x_3d
        for block in self.pressure_path:
            pressure_features = block(pressure_features)
            
        # Combine features
        features = torch.cat([pressure_features, surface_features.unsqueeze(2)], dim=2)
        
        # Transform to latent space
        latent = self.to_latent(features)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)
            
        return latent

