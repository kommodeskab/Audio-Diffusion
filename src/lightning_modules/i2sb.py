from src.lightning_modules import BaseLightningModule
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch

class Scheduler:
    def __init__(
        self,
        num_steps : int,
        min_beta : float,
        max_beta : float,
        ):
        self.num_steps = num_steps
        self.timesteps = torch.arange(num_steps + 1)
        self.delta_t = 1 / num_steps
        
        first_betas = torch.linspace(min_beta, max_beta, num_steps // 2 + 1)
        last_betas = torch.flip(first_betas, [0])
        last_betas = last_betas[1:] if num_steps % 2 == 0 else last_betas
        self.betas = torch.cat([first_betas, last_betas])
        
        self.sigmas_2 = torch.cumsum(self.betas, 0)
        self.sigmas_2_bar = torch.flip(self.sigmas_2, [0])
        
    def shape_for_constant(self, shape : tuple[int]) -> tuple[int]:
        return [-1] + [1] * (len(shape) - 1) 
    
    def reshape_constant(self, constants : list[Tensor], shape : tuple[int]) -> list[Tensor]:
        return [constant.view(self.shape_for_constant(shape)) for constant in constants]     
 
    def sample_timestep(self, batch_size : int) -> Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,))
    
    def sample_posterior(self, x0 : Tensor, x1 : Tensor) -> tuple[Tensor, Tensor]:
        shape = x0.shape
        batch_size = shape[0]
        timesteps = self.sample_timestep(batch_size)
        sigmas_2 = self.sigmas_2[timesteps]
        sigmas_2_bar = self.sigmas_2_bar[timesteps]
        mu_1, mu_2, sigma = self.gaussian_product(sigmas_2_bar, sigmas_2)
        mu_1, mu_2, sigma = self.reshape_constant([mu_1, mu_2, sigma], shape)
        mu = mu_1 * x0 + mu_2 * x1
        std = torch.sqrt(sigma)
        return mu + std * torch.randn_like(mu), timesteps
        
    def gaussian_product(self, v1 : Tensor, v2 : Tensor) -> tuple[Tensor, Tensor]:
        mu_1 = v1 / (v1 + v2)
        mu_2 = v2 / (v1 + v2)
        sigma = v1 * v2 / (v1 + v2)
        
        return mu_1, mu_2, sigma

class I2SB(BaseLightningModule):
    def __init__(
        self,
        model : nn.Module,
        num_steps : int,
        min_beta : float,
        max_beta : float,
        ):
        super().__init__()
        self.model = model
        self.scheduler = Scheduler(
            num_steps=num_steps,
            min_beta=min_beta,
            max_beta=max_beta,
        )
        
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)
    
    def _common_step(self, batch : tuple[Tensor, Tensor]) -> Tensor:
        x0, x1 = batch
        xt, timesteps = self.scheduler.sample_posterior(x0, x1)
        x0_hat = self(xt, timesteps)
        loss = F.mse_loss(x0, x0_hat, reduction='mean')
        return {
            "loss": loss,
        }
    
    def n_to_tensor(self, n : int, batch_size : int) -> Tensor:
        return torch.full((batch_size,), n, dtype=torch.long)
    
    @torch.no_grad()
    def sample(self, x : Tensor, num_inference_steps : int | None = None) -> Tensor:
        batch_size = x.shape[0]
        
        if num_inference_steps is None:
            timesteps = self.scheduler.timesteps.tolist()
        else:
            timesteps = torch.linspace(
                0, 
                self.scheduler.num_steps + 1, 
                num_inference_steps, 
                dtype=torch.long
                ).tolist()
        
        timesteps = timesteps[::-1]
        
        ns = timesteps[1:]
        ns_plus_one = timesteps[:-1]
        
        for n, n_plus_one in zip(ns, ns_plus_one):
            sigmas_2 = self.scheduler.sigmas_2
            alfa_2 = sigmas_2[n_plus_one] - sigmas_2[n]
            sigma_2 = sigmas_2[n]
            n_tensor = self.n_to_tensor(n_plus_one, batch_size)
            x0_hat = self(x, n_tensor)
            mu_1, mu_2, sigma = self.scheduler.gaussian_product(alfa_2, sigma_2)
            mu = mu_1 * x0_hat + mu_2 * x
            std = torch.sqrt(sigma)
            x = mu + std * torch.randn_like(mu)
        
        return x
                
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "loss/val"}