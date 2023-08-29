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
    epsilon = 1e-10  # Small constant to avoid division by zero
    gcc_phat_ij = gcc_ij / (np.abs(gcc_ij) + epsilon)
    if ifft:
        gcc_phat_ij = np.fft.irfft(gcc_phat_ij)
        if abs:
            gcc_phat_ij = np.abs(gcc_phat_ij)

        gcc_phat_ij = np.concatenate((gcc_phat_ij[len(gcc_phat_ij) // 2:],
                                      gcc_phat_ij[:len(gcc_phat_ij) // 2]))

    return gcc_phat_ij
    
def cross_correlation(signals, sr, window_size, stride, plot_peaks=False, n_central_bins=64, output_path=""):
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
    
    signals= signals[:, :15000]   
    n_signals, n_samples = signals.shape
    if n_signals < 2:
        raise ValueError("At least two signals must be provided.")
    
    peak_counts =0
    trimmed_corrs=[]
    for i in range(n_signals):
        for j in range(i, n_signals):
            if i == j:
                continue
            for k in range(0, n_samples-window_size+1, stride):
                window_signals=signals[:, k:k+window_size]
                corr= gcc_phat(window_signals[i], window_signals[j], abs=True, ifft=True, n_dft_bins=None)
                # Plot correlation in the first column,
                threshold=max(corr)/1.2
                peaks, _ = find_peaks(corr, height=threshold)
                peak_counts += len(peaks)
                plt.close('all')
                trimmed_corr = corr
                trimmed_corrs.append( torch.cat((torch.tensor(window_signals[0]), torch.tensor(window_signals[1]), torch.tensor(trimmed_corr))))
                
    matrix=torch.stack(trimmed_corrs)
    return matrix


#CREATE DATASET


class SourceCountingDataset(Dataset):
    def __init__(self, samples_dir, window_size, stride):
        self.samples_dir = samples_dir
        self.sample_folders = sorted(os.listdir(os.path.join(samples_dir, "samples")))
        self.window_size= window_size
        self.stride= stride

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, index):
        sample_path_to_metadata = os.path.join(self.samples_dir, "metadata.json")
        with open(sample_path_to_metadata, "r") as f:
            metadata = json.load(f)
        num_sources = torch.tensor(metadata[index]["n_sources"])
        signals = os.path.join(self.samples_dir, "samples", str(index))
        gcc_phat= cross_correlation(signals, 16000, self.window_size, self.stride, True, 64, "")
        # num_sources = torch.nn.functional.one_hot(num_sources.clone().detach(), num_classes=4).float()
        # Generate the target tensor with shape [25, 28, 4] and fill it with num_sources
        return gcc_phat, num_sources
                                      
   

# LOAD DATASET
class SourceCountingDataLoader(DataLoader):
    def __init__(self, dir, window_size, stride, batch_size=32, shuffle=False, num_workers=5):
        dataset = SourceCountingDataset(dir, window_size, stride)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False, num_workers=num_workers)

if __name__ == "__main__":
    signals = "/datasett/train/samples/0"
    #signals = os.path.join(self.samples_dir, metadata[index]["signals_dir"])
    gcc_phat= cross_correlation(signals, 16000, 1024, 512, True, 64, "")