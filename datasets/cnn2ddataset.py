import os
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import json
from scipy.signal import find_peaks
import numpy as np
import pytorch_lightning as pl
from scipy.signal import correlate, correlation_lags
import torchvision.transforms as transforms



import torchaudio
def plot_stft(signals, sr, index=""):
    if isinstance(signals, str):
        if os.path.isfile(signals):
            signals, sr = torchaudio.load(signals)
        elif os.path.isdir(signals):
            signals_dir = signals
            signals = []
            for file in os.listdir(signals_dir):
                if file.endswith(".wav"):
                    signal, sr = torchaudio.load(os.path.join(signals_dir, file))
                    signals.append(signal)
            if len(signals) == 1:
                signals = signals[0]
            else:
                signals = torch.stack(signals).transpose(0, 1)
        else:
            raise ValueError("The path provided is neither a file nor a directory.")
    elif isinstance(signals, np.ndarray):
        signals = torch.tensor(signals)
    elif isinstance(signals, torch.Tensor):
        pass  
    else:
        raise TypeError("The signals provided must be either a path to a file, a numpy array, or a PyTorch tensor.")
    stft = torch.stft(signals[0], 1024, hop_length= 512, onesided=True, return_complex=True)
    stft= stft.abs()
    stft= stft[:, :90]*10
    return stft
   



#Creating Dataset


class CNN2DDataset(Dataset):
    def __init__(self, samples_dir, window_size, stride):
        self.samples_dir = samples_dir
        self.sample_folders = sorted(os.listdir(os.path.join(samples_dir, "samples")))
        self.window_size= window_size
        self.stride= stride
        sample_path_to_metadata = os.path.join(self.samples_dir, "metadata.json")
        with open(sample_path_to_metadata, "r") as f:
            self.metadata = json.load(f)
    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, index):
        num_sources = torch.tensor(self.metadata[index]["n_sources"])
        signals = os.path.join(self.samples_dir, "samples", str(index))
        stft= plot_stft(signals, 16000, str(index))
        
        num_sources = torch.nn. functional.one_hot(num_sources.clone().detach(), num_classes=3).float()
        return stft, num_sources
                                      
   

# LOAD DATASET
class CNN2DDataLoader(DataLoader):
    def __init__(self, dir, window_size, stride, batch_size=32, shuffle=False, num_workers=0):
        dataset = CNN2DDataset(dir, window_size, stride)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False, num_workers=num_workers)

