import torch
from src.dataset.basedataset import BaseDataset
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import GaussianBlur

class BlurImageDataset(BaseDataset):
    def __init__(
        self, 
        dataset : Dataset,
        kernel_size : int = 5,
        ):
        """
        Returns a dataset that blurs the images in the given dataset.
        """
        
        super().__init__()
        self.dataset = dataset
        self.blur = GaussianBlur(kernel_size = kernel_size)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor]:
        """
        Returns a tuple of the original image and the blurred image.
        """
        original_image = self.dataset[idx]
        blurred_image = self.blur(original_image)
        return original_image, blurred_image
    
class RandomBoxDataset(BaseDataset):
    def __init__(
        self,
        dataset : Dataset,
        box_size : int = 10,
        ):
        """
        Returns a dataset that places a random box onto the image.
        """
        
        super().__init__()
        self.dataset = dataset
        self.box_size = box_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor]:
        """
        Returns a tuple of the original image and the image with a random box.
        """
        original_image = self.dataset[idx]
        image_with_box = original_image.clone()
        x = torch.randint(0, original_image.shape[-1] - self.box_size, (1,)).item()
        y = torch.randint(0, original_image.shape[-2] - self.box_size, (1,)).item()
        image_with_box[:, y : y + self.box_size, x : x + self.box_size] = 1
        return original_image, image_with_box