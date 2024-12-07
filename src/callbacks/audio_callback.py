import torch
from pytorch_lightning import Callback, Trainer
from src.dataset.audio import NoisySpeech
from src.lightning_modules import I2SB
from .utils import get_batch_from_dataset
import matplotlib.pyplot as plt
from torch import Tensor
import wandb

class LogAudio(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : I2SB):
        val_dataset : NoisySpeech = pl_module.val_dataset
        self.sample_rate = val_dataset.dataset.sample_rate
        self.batch = get_batch_from_dataset(val_dataset, batch_size=1)
        logger = pl_module.logger
        
        x0, x1 = self.batch
        for i in range(x0.shape[0]):
            clean, noisy = x0[i].flatten(), x1[i].flatten()
            logger.log_audio(
                f"Original_{i}",
                [clean, noisy],
                step=0,
                sample_rate = [self.sample_rate] * 2
            )
        
    def on_validation_epoch_end(self, trainer : Trainer, pl_module : I2SB):
        logger = pl_module.logger
        
        x0, x1 = self.batch
        x0 = x0.to(pl_module.device)
        x1 = x1.to(pl_module.device)
        
        pl_module.eval()
        trajectory = pl_module.sample(x1, return_trajectory=True).detach().cpu()
        pl_module.train()
        
        n_steps = 5
        num_samples = trajectory.shape[1]
        timesteps = torch.linspace(0, pl_module.scheduler.num_steps, n_steps, dtype=torch.long)
        samples = trajectory[timesteps, :num_samples]
        for i in range(num_samples):
            sample = samples[:, i]
            sample = sample.flatten()
            logger.log_audio(
                f"Sample_{i}",
                [sample],
                step=trainer.global_step,
                sample_rate=[self.sample_rate]
            )