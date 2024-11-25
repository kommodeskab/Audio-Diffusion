import os
files = os.listdir(os.path.dirname(__file__))
if 'data_path.txt' not in files:
    raise FileNotFoundError('data_path.txt not found in dataset folder. Please create a data_path.txt file with the path to the data folder.')
from .emnist import EMNIST, EMNISTNoLabel
from .celeba import CelebADataset
from .basedataset import BaseDataset, ImageDataset