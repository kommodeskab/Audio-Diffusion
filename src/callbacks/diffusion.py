from pytorch_lightning import Callback, Trainer
from src.lightning_modules import I2SB
from .utils import get_batch_from_dataset
import matplotlib.pyplot as plt
from torch import Tensor
import wandb

def format_batch(batch : Tensor) -> Tensor:
    batch = batch * 0.5 + 0.5
    return batch.permute(0, 2, 3, 1).detach().cpu()

def plot_samples(x : Tensor) -> plt.Figure:
    x = format_batch(x)
    cmap = None if x.shape[-1] == 3 else "gray"
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(x[i * 4 + j], cmap=cmap)
            axs[i, j].axis("off")
    return fig

class PlotSamplesCB(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_start(self, trainer : Trainer, pl_module : I2SB):
        val_dataset = pl_module.val_dataset
        self.batch = get_batch_from_dataset(val_dataset, batch_size=16)
        
        x0, x1 = self.batch
        fig_1 = plot_samples(x0)
        fig_2 = plot_samples(x1)
        
        pl_module.logger.log_image(
            "Original samples",
            [wandb.Image(fig_1), wandb.Image(fig_2)],
            step=0
        )
        
        plt.close(fig_1)
        plt.close(fig_2)
        
    def on_validation_epoch_end(self, trainer : Trainer, pl_module : I2SB):
        _, x1 = self.batch
        x0_hat = pl_module.sample(x1)
        
        fig = plot_samples(x0_hat)
        
        pl_module.logger.log_image(
            "Generated samples",
            [wandb.Image(fig)],
            step=pl_module.global_step
        )