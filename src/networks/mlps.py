import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
class ImageMLPEncoder(MLP):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        output_dim: int,
        **kwargs,
    ):
        super().__init__(height * width * channels, output_dim, **kwargs)
        self.height = height
        self.width = width
        self.channels = channels
        self.mu_linear = nn.Linear(output_dim, output_dim)
        self.logvar_linear = nn.Linear(output_dim, output_dim)
        
    def forward(self, x : Tensor) -> Tensor:
        x = x.view(-1, self.height * self.width * self.channels)
        x = super().forward(x)
        x = F.relu(x)
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)
        return mu, logvar
    
class ImageMLPDecoder(MLP):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        input_dim: int,
        **kwargs,
    ):
        super().__init__(input_dim, height * width * channels, **kwargs)
        self.height = height
        self.width = width
        self.channels = channels
        
    def forward(self, x : Tensor) -> Tensor:
        x = super().forward(x)
        return x.view(-1, self.channels, self.height, self.width)

class Image2ImageMLP(MLP):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        input_output_dim = height * width * channels
        super().__init__(
            input_dim = input_output_dim,
            output_dim = input_output_dim,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
        )
        
    def forward(self, x : Tensor) -> Tensor:
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        x = super().forward(x)
        x = x.view(original_shape)
        return x
        
    
if __name__ == "__main__":
    encoder = Image2ImageMLP(28, 28, 1)
    x = torch.randn(10, 1, 28, 28)
    y = encoder(x)
    print(y.shape)