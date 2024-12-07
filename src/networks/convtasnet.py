import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable

_FUN = Callable[[Tensor], Tensor]

class ConditionalConvTasNet(nn.Module):
    def __init__(self, 
                 N=512,        # Number of filters in encoder/decoder
                 L=20,         # Length of the filters (in samples)
                 B=128,        # Number of channels in bottleneck layer
                 H=512,        # Number of channels in convolutional blocks
                 P=3,          # Kernel size in convolutional blocks
                 X=8,          # Number of convolutional blocks in a repeat
                 R=3,          # Number of repeats
                 cond_dim=128, # Dimensionality of the condition embedding
                 num_sources=1 # Number of output sources (1 for denoising)
                 ):
        super().__init__()
        self.N = N
        self.L = L
        self.num_sources = num_sources
        
        # Encoder
        self.encoder : _FUN = nn.Conv1d(1, N, kernel_size=L, stride=L//2, bias=False)
        
        # Timestep embedding
        self.timestep_embedding : _FUN = nn.Sequential(
            nn.Linear(1, cond_dim),  # Embed the timestep
            nn.ReLU(),
            nn.Linear(cond_dim, N)  # Match encoder feature dimension
        )
        
        # Separation network
        self.bottleneck = nn.Conv1d(N, B, 1, bias=False)
        tcn_blocks = []
        for r in range(R):
            for x in range(X):
                tcn_blocks.append(TemporalConvBlock(B, H, P))
        self.tcn = nn.Sequential(*tcn_blocks)
        
        self.mask_conv : _FUN = nn.Conv1d(B, N * num_sources, 1, bias=False)
        
        # Decoder
        self.decoder : _FUN = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)
        
    def forward(self, x : Tensor, t : Tensor):
        """
        x: (batch, channels, sequence_length) - input audio
        t: (batch,) - timesteps for conditioning
        """
        
        latent = self.encoder(x) 
        
        t_embed = self.timestep_embedding(t.float().unsqueeze(-1)) 
        t_embed = t_embed.unsqueeze(-1).expand_as(latent)  
        latent = latent + t_embed  
        
        bottlenecked = self.bottleneck(latent)
        tcn_out = self.tcn(bottlenecked)
        
        masks = self.mask_conv(tcn_out)
        masks = masks.view(masks.size(0), self.num_sources, self.N, -1)
        masks = torch.sigmoid(masks)
        
        masked = latent.unsqueeze(1) * masks
        
        output = self.decoder(masked.view(-1, self.N, masked.size(-1)))
        output = output.view(masks.size(0), self.num_sources, -1)
        
        return output

class TemporalConvBlock(nn.Module):
    def __init__(self, B, H, P):
        super().__init__()
        self.conv1 = nn.Conv1d(B, H, kernel_size=1, bias=False)
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(H)
        
        self.depthwise_conv = nn.Conv1d(H, H, kernel_size=P, padding=(P-1)//2, groups=H, bias=False)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.BatchNorm1d(H)
        
        self.conv2 = nn.Conv1d(H, B, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm1d(B)
        
    def forward(self, x : Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.norm1(x)
        
        x = self.depthwise_conv(x)
        x = self.prelu2(x)
        x = self.norm2(x)
        
        x = self.conv2(x)
        x = self.norm3(x)
        
        return x + residual