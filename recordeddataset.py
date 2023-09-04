import os
import json
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import soundfile as sf
import json
from scipy.signal import find_peaks
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
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

import matplotlib.pyplot as plt

def cross_correlation(signal1, signal2, sr, plot_peaks=False, n_central_bins=64, output_path=""):
    signals=[]
    signals.append(signal1)
    signals.append(signal2)
    signals= np.stack(signals).transpose(0, 1)
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
            threshold=max(corr)/1.2
            peaks, _ = find_peaks(corr, height=threshold)
            peak_counts += len(peaks)
    central_start = len(corr)//2
    trimmed_corr = corr[central_start-200:central_start+200]
    plt.figure(figsize=(12, 6))
    plt.plot(trimmed_corr)
    plt.title("Trimmed Cross-Correlation")
    plt.xlabel("Sample")
    plt.ylabel("Correlation")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    return trimmed_corr


def preprocess_audio(dir1, dir2):
    audio, sample_rate= sf.read(dir1)
    audio2, sample_rate2= sf.read(dir2)
    sample_duration=3
    num_samples= int(len(audio)/(sample_duration*sample_rate))
    samples=[]
    samples2=[]
    for i in range(num_samples):
        samples.append(audio[i*sample_rate*sample_duration:(i+1)*sample_rate*sample_duration])
        samples2.append(audio2[i*sample_rate*sample_duration:(i+1)*sample_rate*sample_duration])
    return samples, samples2

class SourceCountingDataset(Dataset):
    def __init__(self, sample_dir):
        self.sample_dir = "output_mic0.wav"
        self.sample_dir2 = "output_mic1.wav"
        self.samples1, self.samples2=preprocess_audio(self.sample_dir, self.sample_dir2)
       
    def __len__(self):
        return len(self.samples1)

    def __getitem__(self, index):
        signal1 = self.samples1[index]
        signal2= self.samples2[index]
        gcc_phat= cross_correlation(signal1, signal2, 16000, True, 64, "")
        gcc_phat_tensor = torch.tensor(gcc_phat)  # Convert to PyTorch tensor
        #Convert num sources to one hot encoding
        num_sources = torch.tensor([0,1,0])
        return gcc_phat_tensor, num_sources
   


# LOAD DATASET
class SourceCountingDataLoader(DataLoader):
    def __init__(self, dir, batch_size=32, shuffle=True, num_workers=5):
        dataset = SourceCountingDataset(dir)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False, num_workers=num_workers)



import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
#Define the MLP model 
import torch.nn.init as init
import torchvision.ops as ops


def custom_accuracy(predictions, targets):
    differences = torch.abs(predictions - targets)
    max_difference = torch.max(differences)
    normalized_differences = differences / max_difference
    accuracy = 1.0 - normalized_differences
    accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
    mean_accuracy = torch.mean(accuracy)
    return mean_accuracy


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(400, 64), 
            nn.ReLU(),
            nn.Linear(64, 16), 
            nn.ReLU(),  
            nn.Dropout(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )   
        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         # init.xavier_normal_(layer.weight)
        #         # init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        #         init.orthogonal_(layer.weight)
        #         if layer.bias is not None:
        #             init.constant_(layer.bias, 0.0)
    def forward(self, x):
        x=x.to(torch.float32)
        return self.layers(x)

#Define the lightning module using MLP model
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss *10

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class MLPModule(pl.LightningModule):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.model=MLP()
        self.loss=nn.BCELoss()
        self.predictions=[]
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        predicted_classes = torch.argmax(y_hat, dim=1)
        targets= torch.argmax(y, dim=1)

        correct_predictions = (predicted_classes==targets).float()
        accuracy = correct_predictions.mean()
        self.log('val_accuracy', accuracy, prog_bar=True)

        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)
        predicted_classes = torch.argmax(y_hat, dim=1)
        targets= torch.argmax(y, dim=1)
        self.predictions.append(predicted_classes)
        correct_predictions = (predicted_classes==targets).float()
        accuracy = correct_predictions.mean()

        # accuracy= custom_accuracy(torch.round(prediction), y.float())
        self.log('test_accuracy', accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005, weight_decay=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler

        return [optimizer], [scheduler]

from omegaconf import DictConfig
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def test_model(cfg: DictConfig):
    mlp_model=MLPModule.load_from_checkpoint(cfg.checkpoint_path)
    test_data_loader = SourceCountingDataLoader(cfg.test, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    trainer = pl.Trainer()
    trainer.test(model=mlp_model, dataloaders=test_data_loader)
    predictions= mlp_model.predictions
    print(predictions)
    print("Model tested")

if __name__ == "__main__":
    test_model()