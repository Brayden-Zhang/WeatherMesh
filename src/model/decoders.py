import torch
import torch.nn as nn
from einops import rearrange

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_channels, additional_2d_channels):
        super(Decoder, self).__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8), num_layers=6
        )
        self.surface_deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels + additional_2d_channels, kernel_size=3, stride=1, padding=1)
        )
        self.pressure_deconv = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, input_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, latent_space):
        # Reshape from (B, D, H, W, C) to (B, (D*H*W), C)
        B, D, H, W, C = latent_space.shape
        latent_space = rearrange(latent_space, 'b d h w c -> b (d h w) c')
        
        transformed = self.transformer(latent_space)
        
        # Reshape back to (B, D, C, H, W)
        transformed = rearrange(transformed, 'b (d h w) c -> b d c h w', d=D, h=H, w=W)
        
        # Split the combined tensor back into surface and pressure features
        surface_features, pressure_features = torch.split(transformed, [H, D-H], dim=2)
        
        # Ensure the surface features have the shape (B, C, H, W)
        surface_features = rearrange(surface_features, 'b d c h w -> b c h w')
        
        # Decode the features
        surface_output = self.surface_deconv(surface_features)
        pressure_output = self.pressure_deconv(pressure_features)
        
        return surface_output, pressure_output
    

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

