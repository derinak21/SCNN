import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl

# Define the RNN model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        breakpoint()
        self.rnn = nn.RNN(input_size=200, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        
    def forward(self, x):
        x = x.to(torch.float32)
        #convert h0 to torch.float32
        h0 = torch.zeros(1, 64)
        h0 = h0.to(torch.float32)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out)  # Take the last time step's output
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Define the lightning module using RNN model
class RNNModule(pl.LightningModule):
    def __init__(self):
        super(RNNModule, self).__init__()
        self.model = RNN()
        self.loss = nn.MSELoss()
          
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32)
        y_hat = self(x)
        loss = self.loss(y_hat, y.float()) 
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.to(torch.float32)
        loss = self.loss(y_hat, y.float())
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.to(torch.float32)
        loss = self.loss(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)
        predicted_labels = (y_hat >= 0.5).to(torch.float32)  # Convert to binary predictions
        accuracy = (predicted_labels == y).sum().item() / len(y)
        self.log('test_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
