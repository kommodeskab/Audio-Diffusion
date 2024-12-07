from src.lightning_modules import BaseLightningModule
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch
import math
from typing import Literal

class Scheduler:
    def __init__(
        self,
        num_steps : int,
        min_beta : float,
        max_beta : float,
        T : float | None = None
        ):
        self.device = "cpu"
        self.num_steps = num_steps
        self.timesteps = torch.arange(num_steps + 1)
        
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
        
        self.sigmas_2 = torch.cumsum(self.betas, 0)
        self.sigmas_2_bar = torch.flip(self.sigmas_2, [0])
        
    def to_device(self, device : torch.device):
        self.device = device
        self.sigmas_2 = self.sigmas_2.to(device)
        self.sigmas_2_bar = self.sigmas_2_bar.to(device)
        
    def shape_for_constant(self, shape : tuple[int]) -> tuple[int]:
        return [-1] + [1] * (len(shape) - 1) 
    
    def to_tensor(self, value : int, batch_size : int) -> Tensor:
        return torch.full((batch_size,), value, dtype=torch.long).to(self.device)
    
    def reshape_constants(self, constants : list[Tensor], shape : tuple[int]) -> list[Tensor]:
        return [constant.view(self.shape_for_constant(shape)) for constant in constants]     
 
    def sample_timestep(self, batch_size : int) -> Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,)).to(self.device)
    
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
        sample = mu + std * torch.randn_like(mu)
        return sample, ns, sigmas_2
        
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
        learning_rate : float = 1e-4,
        scheduler_patience : int = 5,
        scheduler_factor : float = 0.5,
        target : Literal["flow", "terminal"] = "flow",
        T : float | None = None
        ):
        self.save_hyperparameters(ignore=["model"])
        super().__init__()
        self.model = model
        self.scheduler = Scheduler(
            num_steps=num_steps,
            min_beta=min_beta,
            max_beta=max_beta,
            T=T
        )
        
    def on_train_start(self):
        self.scheduler.to_device(self.device)
        return super().on_train_start()
        
    def forward(self, x : Tensor, timesteps : Tensor) -> Tensor:
        return self.model(x, timesteps)
    
    def predict_x0(self, xt : Tensor, timesteps : Tensor) -> Tensor:
        prediction = self(xt, timesteps)
        
        if self.hparams.target == "flow":
            sigmas = torch.sqrt(self.scheduler.sigmas_2[timesteps])
            sigmas_reshaped, = self.scheduler.reshape_constants([sigmas], xt.shape)
            x0_hat = xt - prediction * sigmas_reshaped
            
        elif self.hparams.target == "terminal":
            x0_hat = prediction
            
        return x0_hat
    
    def _common_step(self, batch : tuple[Tensor, Tensor]) -> Tensor:
        x0, x1 = batch
        xt, timesteps, sigmas_2 = self.scheduler.sample_posterior(x0, x1)
        
        if self.hparams.target == "flow":
            sigmas = torch.sqrt(sigmas_2)
            sigmas_reshaped, = self.scheduler.reshape_constants([sigmas], x0.shape)
            target = (xt - x0) / sigmas_reshaped
            
        elif self.hparams.target == "terminal":
            target = xt
        
        prediction = self(xt, timesteps)
        loss = F.mse_loss(target, prediction, reduction='mean')
        return {
            "loss": loss,
        }
    
    @torch.no_grad()
    def sample(
        self, 
        x : Tensor, 
        num_inference_steps : int | None = None,
        return_trajectory : bool = False,
        clamp : bool = False,
        ) -> Tensor:
        x = x.to(self.device)
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
        
        trajectories = [x]
        
        for n, n_plus_one in zip(ns, ns_plus_one):
            sigmas_2 = self.scheduler.sigmas_2
            alfa_2 = sigmas_2[n_plus_one] - sigmas_2[n]
            sigma_2 = sigmas_2[n]
            n_tensor = self.scheduler.to_tensor(n_plus_one, batch_size)
            x0_hat = self.predict_x0(x, n_tensor)
            mu_1, mu_2, sigma = self.scheduler.gaussian_product(alfa_2, sigma_2)
            mu = mu_1 * x0_hat + mu_2 * x
            std = torch.sqrt(sigma)
            x = mu + std * torch.randn_like(mu)
            trajectories.append(x)
        
        trajectories = torch.stack(trajectories, dim=0)
        
        if clamp:
            trajectories = torch.clamp(trajectories, -1, 1)
            
        return trajectories if return_trajectory else trajectories[-1]

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='min', 
            factor=self.hparams.scheduler_factor, 
            patience=self.hparams.scheduler_patience
            )
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "loss/val"}