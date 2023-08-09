import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl

#Define the MLP model 
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),   
            nn.Linear(32, 1),            
            nn.Sigmoid()

        )
        self.loss = nn.BCELoss()
    def forward(self, x):
        x=x.to(torch.float32)
        return self.layers(x)

#Define the lightning module using MLP model

class MLPModule(pl.LightningModule):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.model=MLP()
        self.loss=nn.BCELoss()
          
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float()) 
        print(loss)
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
        print(accuracy)
        return {'test_accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
