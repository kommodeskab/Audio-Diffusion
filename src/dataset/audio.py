from torch.utils.data import Dataset, ConcatDataset
import torchaudio
import os
import torch
import random
from typing import Tuple
from src.dataset.basedataset import BaseDataset
from torch import Tensor

class AudioCrawler(Dataset):
    def __init__(self, path : str):
        super().__init__()
        self.path = path
        self.files = [f for f in os.listdir(path) if f.endswith(".wav")]
        
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
            start = random.randint(0, seq_length - new_seq_length)
            waveform = waveform[:, start:start+new_seq_length]
        else:
            padding_size = new_seq_length - seq_length
            left_padding = random.randint(0, padding_size)
            right_padding = padding_size - left_padding
            waveform = torch.nn.functional.pad(waveform, (left_padding, right_padding))
        
        return waveform
    
class Noise(ConcatFormattedAudio, BaseDataset):
    def __init__(
        self,
        audio_length : int = 2,
        sample_rate : int = 16000,
    ):
        data_path = self.data_path
        paths = [
            f"{data_path}/MS-SNSD/noise_train",
            f"{data_path}/MS-SNSD/noise_test",
            f"{data_path}/FSDNoisy18k/FSDnoisy18k.audio_test",
            f"{data_path}/FSDNoisy18k/FSDnoisy18k.audio_train",
        ]
        datasets = [AudioCrawler(path) for path in paths]
        super().__init__(datasets, audio_length=audio_length, sample_rate=sample_rate)
        
class Clean(ConcatFormattedAudio, BaseDataset):
    def __init__(
        self,
        audio_length : int = 2,
        sample_rate : int = 16000,
    ):
        data_path = self.data_path
        paths = [
            f"{data_path}/MS-SNSD/clean_train",
            f"{data_path}/MS-SNSD/clean_test",
        ]
        datasets = [AudioCrawler(path) for path in paths]
        super().__init__(datasets, audio_length=audio_length, sample_rate=sample_rate)
        
class NoisySpeech(BaseDataset):
    def __init__(
        self,
        audio_length : int = 2,
        sample_rate : int = 16000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.noise = Noise(audio_length=audio_length, sample_rate=sample_rate)
        self.clean = Clean(audio_length=audio_length, sample_rate=sample_rate)
        
    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, idx):
        clean = self.clean[idx]
        noise_idx = random.randint(0, len(self.noise) - 1)
        noise = self.noise[noise_idx]
        # if the noise is all zeros, just generate some random noise
        if noise.abs().sum() < 1e-3:
            print("NoisySpeech: Noise is all zeros, generating random noise")
            noise = torch.randn_like(clean)
        
        alpha = random.uniform(1.0, 2.0)
        noisy_speech = clean + alpha * noise
        
        return clean, noisy_speech
    
class NoisySpeechSpectrogram(NoisySpeech):
    def __init__(
        self, 
        audio_length : int = 2,
        sample_rate : int = 16000,
        ):
        super().__init__(audio_length=audio_length, sample_rate=sample_rate)    
        self.n_fft = 160
        self.hop_length = 160
        self.win_length = 160
        
    def to_spectrogram(self, audio : Tensor) -> Tensor:
        spectrogram = torch.stft(
            audio.squeeze(1), # remove channel
            self.n_fft,
            self.hop_length,
            self.win_length,
            return_complex=True
        ).unsqueeze(1)
        return spectrogram
    
    def to_audio(self, spectrogram : Tensor) -> Tensor:
        audio = torch.istft(
            spectrogram.squeeze(1), 
            self.n_fft, 
            self.hop_length, 
            self.win_length
            ).unsqueeze(1)
    
    def __getitem__(self, idx):
        clean, noisy_speech = super().__getitem__(idx)
        clean_mel = self.audio_to_mel(clean)
        noisy_speech_mel = self.audio_to_mel(noisy_speech)
        return clean_mel, noisy_speech_mel