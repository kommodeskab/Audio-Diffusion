from torch.utils.data import Dataset
import hashlib
from torchvision import transforms
from .utils import get_data_path
from torch import Tensor

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @property
    def unique_identifier(self):
        attr_str = str(vars(self))
        return hashlib.md5(attr_str.encode()).hexdigest()
    
    @property
    def data_path(self):
        return get_data_path()
    
class ImageDataset(BaseDataset):
    def __init__(
        self,
        dataset : Dataset,
        img_size : int,
        augment : bool = False,
        horizontal_flip : bool = False,
    ):
        """
        The base image dataset class.
        Enables data augmentation and resizing for extended dataset size.
        """
        super().__init__()
        self.dataset = dataset
        
        if augment:
            p_flip = 0.5 if horizontal_flip else 0.0
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomHorizontalFlip(p_flip),
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.9, 1.0)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx : int) -> Tensor:
        img : Tensor = self.dataset[idx]
        img = self.transform(img).clamp(-1, 1)
        return img