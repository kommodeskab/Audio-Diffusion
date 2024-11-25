from torch.utils.data import Dataset, ConcatDataset
import torchaudio
import os
import torch
from typing import Tuple

class AudioCrawler(Dataset):
    def __init__(self, path : str):
        super().__init__()
        self.path = path
        self.files = os.listdir(path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        filename = self.files[idx]
        waveform, sample_rate = torchaudio.load(os.path.join(self.path, filename))
        return waveform, sample_rate
    
class ConcatFormattedAudio(ConcatDataset):
    def __init__(
        self, 
        datasets : list[Dataset],
        audio_length : int = 2,
        sample_rate : int = 16000,
        ):
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        super().__init__(datasets)
        
    def __getitem__(self, idx):
        waveform, old_sample_rate = super().__getitem__(idx)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.transforms.Resample(old_sample_rate, self.sample_rate)(waveform)
        
        seq_length = waveform.shape[1]
        new_seq_length = self.audio_length * self.sample_rate
        if seq_length > new_seq_length:
            waveform = waveform[:, :new_seq_length]
        else:
            padding_size = new_seq_length - seq_length
            waveform = torch.nn.functional.pad(waveform, (0, padding_size))
        
        return waveform