import pytorch_lightning as pl
import torch.utils
from torch.utils.data import DataLoader, random_split, Dataset

class BaseDM(pl.LightningDataModule):
    def __init__(
        self,
        dataset : Dataset,
        val_dataset : Dataset = None,
        train_val_split : float = 0.95,
        batch_size : int = 32,
        num_workers: int = 4,
        ):
        """
        A base data module for datasets. 
        It takes a dataset and splits into train and validation (if val_dataset is None).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "val_dataset"])
        
        self.dataset = dataset        
        if val_dataset is None:
            self.train_dataset, self.val_dataset = random_split(dataset, [train_val_split, 1 - train_val_split])
        else:
            self.train_dataset, self.val_dataset = dataset, val_dataset
        
    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset, 
            shuffle = True, 
            drop_last = True,
            persistent_workers = True,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers
            )
        
    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_dataset, 
            shuffle = False, 
            drop_last = True,
            persistent_workers = True,
            batch_size = self.hparams.batch_size, 
            num_workers = self.hparams.num_workers
            )