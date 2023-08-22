import os
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import json
from scipy.signal import find_peaks
import numpy as np
import torch.nn as nn


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
            # Plot correlation in the first column,
            corr = gcc_phat(signals[i], signals[j], abs=True, ifft=True, n_dft_bins=None)
            # threshold=max(corr)/1.2
            # peaks, _ = find_peaks(corr, height=threshold)  # You can adjust the height threshold if needed.
            # peak_counts += len(peaks)
            plt.close('all')
    
    central_start = len(corr)//2
    trimmed_corr = corr[central_start-100:central_start+100]
    return trimmed_corr

def process_folder(base_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for dataset_folder in os.listdir(base_folder):  # Loop through all the dataset folders
        dataset_path = os.path.join(base_folder, dataset_folder)
        if os.path.isdir(dataset_path):
            print("Processing", dataset_folder)
            sample_path_to_metadata = os.path.join(dataset_path, "metadata.json")
            with open(sample_path_to_metadata, "r") as f:
                metadata = json.load(f)
            all_gcc_phat_tensors = []
            targets=[]
            subfolder_path = os.path.join(dataset_path, "samples").replace("\\", "/")
            i=0
            for file in os.listdir(subfolder_path):
                targets.append(torch.tensor(len(metadata[i]["source_coordinates"])))
                print(torch.tensor(len(metadata[i]["source_coordinates"])))
                i+=1
                file = os.path.join(subfolder_path, file).replace("\\", "/")
                if os.path.isdir(file):
                    gcc_phat_ = cross_correlation(file, 16000, True, 64, "")
                    gcc_phat_tensor = torch.tensor(gcc_phat_)  # Convert to PyTorch tensor
                    all_gcc_phat_tensors.append(gcc_phat_tensor)
                    del gcc_phat_
                    del gcc_phat_tensor
                # Save the list of tensors to a .pt file
            output_file = os.path.join(output_folder, f"{dataset_folder}.pt")
            combined_data={"gcc_phat":all_gcc_phat_tensors,"targets":targets}
            torch.save(combined_data, output_file)
            print("GCC-PHAT tensors saved to", output_file)


import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config")

def main(cfg: DictConfig):
    process_folder(cfg.base_folder, cfg.output_folder)
  
if __name__ == "__main__":
    main()

    

 