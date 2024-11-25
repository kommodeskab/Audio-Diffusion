from .baselightningmodule import BaseLightningModule
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Union
from pytorch_lightning.utilities import grad_norm

class SimpleVAE(BaseLightningModule):
    def __init__(
        self, 
        encoder : Union[Callable[[Tensor], Tensor], torch.nn.Module],
        decoder : Union[Callable[[Tensor], Tensor], torch.nn.Module],
        latent_dim : int,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def _common_step(self, batch):
        mu, log_var = self.encoder(batch)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        mse, kld = self.loss_function(x_hat, batch, mu, log_var)
        return {
            "loss": mse + kld,
            "mse": mse,
            "kld": kld
        }
        
    def on_before_optimizer_step(self, optimizer):
        encoder_norm : dict[str, Tensor] = grad_norm(self.encoder, norm_type=2)
        decoder_norm : dict[str, Tensor] = grad_norm(self.decoder, norm_type=2)
        self.log("encoder_grad_norm", encoder_norm['grad_2.0_norm_total'])
        self.log("decoder_grad_norm", decoder_norm['grad_2.0_norm_total'])
    
    def reparameterize(self, mu : Tensor, log_var : Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss_function(self, x_hat : Tensor, x : Tensor, mu : Tensor, log_var : Tensor) -> Tensor:
        mse = F.mse_loss(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        batch_size = x.size(0)
        mse /= batch_size
        kld /= batch_size
        return mse, kld
    
    def sample(self, num_samples : int) -> Tensor:
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        return self.decoder(z)
    
    def reconstruct(self, x : Tensor) -> Tensor:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)
        return {"optimizer": optim, "lr_scheduler": scheduler, "monitor": "loss/val"}