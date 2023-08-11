import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl

#Define the CNN model 
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layers=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64, 32), 
            nn.ReLU(),   
            nn.Linear(32, 1),            
            nn.Sigmoid()

        )
        self.loss = nn.BCELoss()
    def forward(self, x):
        x=x.to(torch.float32)
        return self.layers(x)
    

    
class CNN1DModule(pl.LightningModule):
    def __init__(self):
        super(CNN1DModule, self).__init__()
        self.model=CNN1D()
        self.loss=nn.BCELoss()
          
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
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)
        predicted_labels = (y_hat >= 0.5).to(torch.float32)  # Convert to binary predictions
        accuracy = (predicted_labels == y).sum().item() / len(y)
        self.log('test_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
