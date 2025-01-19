import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class NeighborhoodAttention(nn.Module):
    def __init__(
        self,
        dim,
        depth_window=5,
        height_window=7,
        width_window=7,
        heads=8,
        qkv_bias=True,
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.depth_window = depth_window
        self.height_window = height_window
        self.width_window = width_window
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Rotary position embedding
        self.register_buffer(
            "rel_pos_h",
            self._get_rotary_embedding(height_window),
            persistent=False,
        )
        self.register_buffer(
            "rel_pos_w",
            self._get_rotary_embedding(width_window),
            persistent=False,
        )
        self.register_buffer(
            "rel_pos_d",
            self._get_rotary_embedding(depth_window),
            persistent=False,
        )
        
    def _get_rotary_embedding(self, window_size):
        coords = torch.arange(window_size)
        coords = coords[:, None] - coords[None, :]
        return coords.float()
        
    def forward(self, x):
        B, D, H, W, C = x.shape
        
        # Pad input for windowing
        pad_d = (self.depth_window - 1) // 2
        pad_h = (self.height_window - 1) // 2
        pad_w = (self.width_window - 1) // 2
        
        x_padded = F.pad(
            x,
            (0, 0, pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
            mode='circular'
        )
        
        # Extract neighborhoods
        neighborhoods = F.unfold(
            x_padded.permute(0, 4, 1, 2, 3),
            kernel_size=(self.depth_window, self.height_window, self.width_window),
            padding=0,
            stride=1
        )
        
        # Reshape for attention
        neighborhoods = neighborhoods.view(
            B, self.heads, C // self.heads,
            self.depth_window * self.height_window * self.width_window,
            D * H * W
        ).permute(0, 1, 4, 3, 2)
        
        # Apply attention
        qkv = self.qkv(neighborhoods).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(
                t,
                'b h p n (heads d) -> b heads p n d',
                heads=self.heads
            ),
            qkv
        )
        
        # Apply rotary embeddings and compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        x = (attn @ v)
        x = rearrange(x, 'b heads p n d -> b p n (heads d)')
        x = self.proj(x)
        
        # Reshape back to original format
        x = x.view(B, D, H, W, C)
        
        return x
