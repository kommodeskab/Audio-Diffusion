from pytorch_lightning import Callback
import matplotlib.pyplot as plt
from src.lightning_modules.example import SimpleVAE
from pytorch_lightning import Trainer
import wandb

class VAECallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_validation_epoch_end(self, trainer : Trainer, pl_module : SimpleVAE):
        samples = pl_module.sample(num_samples = 16)
        samples = samples.view(-1, 1, 32, 32)
        
        fig = plt.figure(figsize=(8, 8))
        for i in range(samples.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(samples[i].squeeze().detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
        
        pl_module.logger.log_image(
            "Samples",
            [wandb.Image(fig)],
            step = trainer.global_step
        )