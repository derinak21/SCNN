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
    gcc_phat_ij = gcc_ij / (np.abs(gcc_ij)+1e-10)

    if ifft:
        gcc_phat_ij = np.fft.irfft(gcc_phat_ij)
        if abs:
            gcc_phat_ij = np.abs(gcc_phat_ij)

        gcc_phat_ij = np.concatenate((gcc_phat_ij[len(gcc_phat_ij) // 2:],
                                      gcc_phat_ij[:len(gcc_phat_ij) // 2]))

    return gcc_phat_ij

import matplotlib.pyplot as plt

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
    signals= signals.T
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

def preprocess_audio(dir1, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio, sample_rate = sf.read(dir1)
    sample_duration = 3
    num_samples = int(len(audio) / (sample_duration * sample_rate))

    # Iterate through samples and save them as separate WAV files
    for i in range(num_samples):
        sample = audio[i * sample_rate * sample_duration : (i + 1) * sample_rate * sample_duration]
        
        # Define the output filename with a ".wav" extension
        output_filename = f"sample_{i}.wav"
        
        # Combine the output directory path and filename
        output_path = os.path.join(output_dir, output_filename)
        
        # Use sf.write to save the sample as a WAV file
        sf.write(output_path, sample, sample_rate)


class SourceCountingDataset(Dataset):
    def __init__(self, sample_dir):
        self.sample_dir = "recording/recording.wav"
        self.samples= preprocess_audio(self.sample_dir, "audio")
    def __len__(self):
        return len(os.listdir("audio"))

    def __getitem__(self, index):
        path= os.path.join("audio", f"sample_{index}.wav")
        gcc_phat= cross_correlation(path, 16000, True, 64, "")
        gcc_phat_tensor = gcc_phat  # Convert to PyTorch tensor
        #Convert num sources to one hot encoding
        if index<=40:
            num_sources = torch.tensor([1.0,0.0,0.0])
        elif index <=80:
            num_sources = torch.tensor([0.0,1.0,0.0])
        else:
            num_sources = torch.tensor([0.0,0.0,1.0])
    
        return gcc_phat_tensor, num_sources
   


# LOAD DATASET
class SourceCountingDataLoader(DataLoader):
    def __init__(self, dir, batch_size=32, shuffle=False, num_workers=0):
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




class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1= nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.pool= nn.MaxPool1d(kernel_size=2)
        self.conv2= nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1= nn.Linear(3200, 128)
        self.fc2= nn.Linear(128, 3)
        self.dropout= nn.Dropout(p=0.4)
        self.softmax= nn.Softmax(dim=1)
    def forward(self, x):
        x=x.to(torch.float32) 
        x= self.pool(torch.relu(self.conv1(x))) 
        x= self.pool(torch.relu(self.conv2(x))) 
        x= x.view(x.shape[0], x.shape[1]*x.shape[2])
        x= torch.relu(self.fc1(x))
        x= self.fc2(x)
        x= self.dropout(x)
        x= self.softmax(x)
        return x
    

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
        
class CNN1DModule(pl.LightningModule):
    def __init__(self):
        super(CNN1DModule, self).__init__()
        self.model = CNN1D()
        lossdict={
            'BCELoss': nn.BCELoss(),
            'MSELoss': nn.MSELoss(),
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
            'FocalLoss': FocalLoss(alpha=0.5, gamma=2, reduction='mean'),
        }
        self.loss=nn.BCELoss()
        self.learning_rate=0.001
        self.weight_decay=0
        self.scheduler=None
        self.predictions=[]
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        predicted_classes = torch.argmax(y_hat, dim=1)
        targets = torch.argmax(y, dim=1)
        correct_predictions = (predicted_classes == targets).float()
        accuracy = correct_predictions.mean()
        self.log('val_accuracy', accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        breakpoint()
        self.log('test_loss', loss, prog_bar=True)
        predicted_classes = torch.argmax(y_hat, dim=1)
        targets = torch.argmax(y, dim=1)
        self.predictions.append(predicted_classes)
        correct_predictions = (predicted_classes == targets).float()
        accuracy = correct_predictions.mean()
        self.log('test_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler
            return [optimizer], [scheduler]


from omegaconf import DictConfig
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def test_model(cfg: DictConfig):
    cnn_model=CNN1DModule.load_from_checkpoint(cfg.checkpoint_path)
    test_data_loader = SourceCountingDataLoader(cfg.test, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.num_workers)
    trainer = pl.Trainer()
    trainer.test(model=cnn_model, dataloaders=test_data_loader)
    predictions= cnn_model.predictions
    print(predictions)
    print("Model tested")

if __name__ == "__main__":
    test_model()