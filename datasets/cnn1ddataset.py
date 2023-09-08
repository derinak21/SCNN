import os
import json
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
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
# GET THE GCC_PHAT OF SIGNALS AND THE NUMBER OF SOURCES
# LOAD THE GCC_PHAT OF SIGNALS


#PREPROCESSING

def gcc_phat(signal1, signal2, abs=True, ifft=True, n_dft_bins=None):
    """Compute the generalized cross-correlation with phase transform (GCC-PHAT) between two signals.

    Parameters
    ----------
    signal1 : np.ndarray
        The first signal to correlate.
    signal2 : np.ndarray
        The second signal to correlate.
    abs : bool
        Whether to take the absolute value of the cross-correlation. Only used if ifft is True.
    ifft : bool
        Whether to use the inverse Fourier transform to compute the cross-correlation in the time domain,
        instead of returning the cross-correlation in the frequency domain.
    n_dft_bins : int
        The number of DFT bins to use. If None, the number of DFT bins is set to n_samples//2 + 1.
    """
    n_samples = len(signal1)

    if n_dft_bins is None:
        n_dft_bins = n_samples // 2 + 1

    signal1_dft = np.fft.rfft(signal1, n=n_dft_bins)
    signal2_dft = np.fft.rfft(signal2, n=n_dft_bins)

    gcc_ij = signal1_dft * np.conj(signal2_dft)
    gcc_phat_ij = gcc_ij / np.abs(gcc_ij)

    if ifft:
        gcc_phat_ij = np.fft.irfft(gcc_phat_ij)
        if abs:
            gcc_phat_ij = np.abs(gcc_phat_ij)

        gcc_phat_ij = np.concatenate((gcc_phat_ij[len(gcc_phat_ij) // 2:],
                                      gcc_phat_ij[:len(gcc_phat_ij) // 2]))

    return gcc_phat_ij
    
def cross_correlation(signals, sr, plot_peaks=False, n_central_bins=64, output_path=""):
    if isinstance(signals, str):
        if os.path.isfile(signals):
            signals, sr = sf.read(signals)
        elif os.path.isdir(signals):
            signals_dir = signals
            signals = []
            for file in os.listdir(signals_dir):
                if file.endswith(".wav"):
                    signal, sr = sf.read(os.path.join(signals_dir, file))
                    signals.append(signal)
            if len(signals) == 1:
                signals = signals[0].T
            else:
                signals = np.stack(signals).transpose(1, 2)
        else:
            raise ValueError("The path provided is neither a file nor a directory.")
    elif not isinstance(signals, np.ndarray):
        raise TypeError("The signals provided must be either a path to a file or a numpy array.")
    
        
    n_signals, n_samples = signals.shape
    if n_signals < 2:
        raise ValueError("At least two signals must be provided.")
    
    peak_counts =0

    for i in range(n_signals):
        for j in range(i, n_signals):
            if i == j:
                continue
            corr = gcc_phat(signals[i], signals[j], abs=True, ifft=True, n_dft_bins=None)
            threshold=max(corr)/1.2
            # if threshold<0.1:
            #     threshold=0.1
            peaks, _ = find_peaks(corr, height=threshold)
            peak_counts += len(peaks)
            is_peak=torch.zeros(len(corr))
            is_peak[peaks]=1
    
    central_start = len(corr)//2
    trimmed_corr = corr[central_start-200:central_start+200]
    trimmed_is_peak= is_peak[central_start-200:central_start+200]
    matrix=[]
    matrix.append(torch.tensor(trimmed_corr))
    matrix.append(trimmed_is_peak)
    matrix= torch.stack(matrix)
    return matrix


#Creating Dataset

class CNN1DDataset(Dataset):
    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        self.sample_folders = sorted(os.listdir(os.path.join(samples_dir, "samples")))
        sample_path_to_metadata = os.path.join(self.samples_dir, "metadata.json")
        with open(sample_path_to_metadata, "r") as f:
            self.metadata = json.load(f)
    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, index):
        num_sources = torch.tensor(self.metadata[index]["n_sources"])
        signals = os.path.join(self.samples_dir, "samples", str(index))
        input= cross_correlation(signals, 16000, True, 64, "")
        num_sources = torch.nn.functional.one_hot(num_sources.clone().detach(), num_classes=3).float()
        return input, num_sources                              
   


# Loading Dataset
class CNN1DDataLoader(DataLoader):
    def __init__(self, dir, window_size, stride, batch_size=32, shuffle=True, num_workers=0):
        dataset = CNN1DDataset(dir)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False, num_workers=num_workers)
        

