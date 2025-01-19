import torch
import torch.nn as nn
from typing import List, Tuple

from model.decoders import WeatherMeshDecoder
from model.encoders import WeatherMeshEncoder
from model.processor import WeatherMeshProcessor

class WeatherMesh(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        processors: List[nn.Module],
        decoder: nn.Module,
        timesteps: List[int]
    ):
        super().__init__()
        self.encoder = WeatherMeshEncoder(
            input_channels_2d=8,
            input_channels_3d=4,
            latent_dim=256,
            n_pressure_levels=25
        )
        self.processors = nn.ModuleList(WeatherMeshProcessor(latent_dim=256) for _ in timesteps)
        self.decoder = WeatherMeshDecoder(
            latent_dim=256,
            output_channels_2d=8,
            output_channels_3d=4,
            n_pressure_levels=25
        )
        self.timesteps = timesteps
        
    def forward(
        self,
        x_2d: torch.Tensor,
        x_3d: torch.Tensor,
        forecast_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode input
        latent = self.encoder(x_2d, x_3d)
        
        # Apply processors for each forecast step
        for _ in range(forecast_steps):
            for processor in self.processors:
                latent = processor(latent)
                
        # Decode output
        surface_out, pressure_out = self.decoder(latent)
        
        return surface_out, pressure_out
