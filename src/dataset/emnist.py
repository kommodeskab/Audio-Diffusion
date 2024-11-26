from torchvision import datasets, transforms
from .basedataset import BaseDataset
from torch import Tensor

class EMNIST(BaseDataset):
    def __init__(self, split : str, img_size : int = 32):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                    transforms.Normalize((0.5), (0.5)),
                ])
        self.emnist_dataset = datasets.EMNIST(
            root=self.data_path,
            split=split,
            download=True,
            transform=transform,
        )
        super().__init__()

    @property
    def targets(self):
        return self.emnist_dataset.targets
        
    def __len__(self):
        return len(self.emnist_dataset)
    
    def __getitem__(self, idx : int) -> Tensor:
        image, _ = self.emnist_dataset[idx]
        image = transforms.functional.rotate(image, 90)
        image = transforms.functional.vflip(image)
        return image