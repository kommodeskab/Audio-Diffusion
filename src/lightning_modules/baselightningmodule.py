import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch.nn.init as init
import torch
from torch import Tensor
from typing import Any
from pytorch_lightning.utilities import grad_norm

class BaseLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def _common_step(self, batch : Any) -> dict[str, Tensor]:
        raise NotImplementedError("Common step not implemented")
    
    def training_step(self, batch : Any, batch_idx : int) -> Tensor:
        loss_dict = self._common_step(batch)
        loss_dict = self._convert_dict_losses(loss_dict, suffix="train")
        self.log_dict(loss_dict, prog_bar = True)
        return loss_dict["loss/train"]
    
    def validation_step(self, batch : Any, batch_idx : int) -> Tensor:
        loss_dict = self._common_step(batch)
        loss_dict = self._convert_dict_losses(loss_dict, suffix="val")
        self.log_dict(loss_dict, prog_bar = True)
        return loss_dict["loss/val"]
    
    def on_before_optimizer_step(self, optimizer):
        grad_norm_dict : dict[str, Tensor] = grad_norm(self, norm_type=2)
        self.log("grad_norm", grad_norm_dict['grad_2.0_norm_total'])
        
    def _convert_dict_losses(self, losses : dict, suffix : str = "", prefix : str = "") -> dict:
        if suffix:
            losses = {f"{k}/{suffix}": v for k, v in losses.items()}
        if prefix:
            losses = {f"{prefix}/{k}": v for k, v in losses.items()}
        return losses
    
    @property
    def logger(self) -> WandbLogger:
        return self.trainer.logger
    
    @staticmethod
    def init_weights(model : nn.Module) -> None:
        """
        Initializes the weights of the forward and backward models  
        using the Kaiming Normal initialization
        """
        @torch.no_grad()
        def initialize(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Apply initialization to both networks
        model.apply(initialize)