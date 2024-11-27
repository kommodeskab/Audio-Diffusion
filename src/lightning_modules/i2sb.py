from src.lightning_modules import BaseLightningModule
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch
import math

class Scheduler:
    def __init__(
        self,
        num_steps : int,
        min_beta : float,
        max_beta : float,
        T : float | None = None
        ):
        self.num_steps = num_steps
        self.timesteps = torch.arange(num_steps + 1)
        self.delta_t = 1 / num_steps
        
        # make symmetric beta schedule
        betas = torch.zeros(num_steps)
        first_betas_len = math.ceil(num_steps / 2)
        betas[:first_betas_len] = torch.linspace(min_beta, max_beta, first_betas_len)
        betas[-first_betas_len:] = torch.flip(betas[:first_betas_len], [0])
        self.betas = torch.cat([torch.zeros(1), betas])
        
        if T is not None:
            self.betas = self.betas / self.betas.sum() * T
        else:
            self.T = self.betas.sum()
        
        self.sigmas_2 = torch.cumsum(self.betas, 0) * self.delta_t
        self.sigmas_2_bar = torch.flip(self.sigmas_2, [0])
        
    def shape_for_constant(self, shape : tuple[int]) -> tuple[int]:
        return [-1] + [1] * (len(shape) - 1) 
    
    def to_tensor(self, value : int, batch_size : int) -> Tensor:
        return torch.full((batch_size,), value, dtype=torch.long)
    
    def reshape_constants(self, constants : list[Tensor], shape : tuple[int]) -> list[Tensor]:
        return [constant.view(self.shape_for_constant(shape)) for constant in constants]     
 
    def sample_timestep(self, batch_size : int) -> Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,))
    
    def sample_posterior(self, x0 : Tensor, x1 : Tensor, n : int | None = None) -> tuple[Tensor, Tensor]:
        shape = x0.shape
        batch_size = shape[0]
        ns = self.to_tensor(n, batch_size) if n is not None else self.sample_timestep(batch_size)
        sigmas_2 = self.sigmas_2[ns]
        sigmas_2_bar = self.sigmas_2_bar[ns]
        mu_1, mu_2, sigma = self.gaussian_product(sigmas_2_bar, sigmas_2)
        mu_1, mu_2, sigma = self.reshape_constants([mu_1, mu_2, sigma], shape)
        mu = mu_1 * x0 + mu_2 * x1
        std = torch.sqrt(sigma)
        return mu + std * torch.randn_like(mu), ns
        
    def gaussian_product(self, v1 : Tensor, v2 : Tensor) -> tuple[Tensor, Tensor]:
        mu_1 = v1 / (v1 + v2)
        mu_2 = v2 / (v1 + v2)
        sigma = v1 * v2 / (v1 + v2)
        
        return mu_1, mu_2, sigma
        
    def shape_for_constant(self, shape : tuple[int]) -> tuple[int]:
        return [-1] + [1] * (len(shape) - 1) 
    
    def to_tensor(self, value : int, batch_size : int) -> Tensor:
        return torch.full((batch_size,), value, dtype=torch.long)
    
    def reshape_constants(self, constants : list[Tensor], shape : tuple[int]) -> list[Tensor]:
        return [constant.view(self.shape_for_constant(shape)) for constant in constants]     
 
    def sample_timestep(self, batch_size : int) -> Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,))
    
    def sample_posterior(self, x0 : Tensor, x1 : Tensor, n : int | None = None) -> tuple[Tensor, Tensor]:
        shape = x0.shape
        batch_size = shape[0]
        ns = self.to_tensor(n, batch_size) if n is not None else self.sample_timestep(batch_size)
        sigmas_2 = self.sigmas_2[ns]
        sigmas_2_bar = self.sigmas_2_bar[ns]
        mu_1, mu_2, sigma = self.gaussian_product(sigmas_2_bar, sigmas_2)
        mu_1, mu_2, sigma = self.reshape_constants([mu_1, mu_2, sigma], shape)
        mu = mu_1 * x0 + mu_2 * x1
        std = torch.sqrt(sigma)
        return mu + std * torch.randn_like(mu), ns
        
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
        T : float | None = None
        ):
        super().__init__()
        self.model = model
        self.scheduler = Scheduler(
            num_steps=num_steps,
            min_beta=min_beta,
            max_beta=max_beta,
            T=T
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
    
    @torch.no_grad()
    def sample(
        self, 
        x : Tensor, 
        num_inference_steps : int | None = None,
        return_trajectory : bool = False
        ) -> Tensor:
        
        batch_size = x.shape[0]
        if num_inference_steps is None:
            timesteps = self.scheduler.timesteps.tolist()
        else:
            timesteps = torch.linspace(
                0, 
                self.scheduler.num_steps + 1, 
                num_inference_steps + 1,  # num_states = num_steps + 1
                dtype=torch.long
                ).tolist()
        
        timesteps = timesteps[::-1]
        
        ns = timesteps[1:]
        ns_plus_one = timesteps[:-1]
        
        trajectories = []
        
        for n, n_plus_one in zip(ns, ns_plus_one):
            sigmas_2 = self.scheduler.sigmas_2
            alfa_2 = sigmas_2[n_plus_one] - sigmas_2[n]
            sigma_2 = sigmas_2[n]
            n_tensor = self.scheduler.to_tensor(n_plus_one, batch_size)
            x0_hat = self(x, n_tensor)
            mu_1, mu_2, sigma = self.scheduler.gaussian_product(alfa_2, sigma_2)
            mu = mu_1 * x0_hat + mu_2 * x
            std = torch.sqrt(sigma)
            x = mu + std * torch.randn_like(mu)
            trajectories.append(x)
        
        trajectories = torch.stack(trajectories, dim=0)
        return trajectories if return_trajectory else trajectories[-1]
                
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "loss/val"}