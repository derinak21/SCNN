import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_

#Define the CNN model 
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1= nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.pool= nn.MaxPool1d(kernel_size=2)
        self.conv2= nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1= nn.Linear(1600, 128)
        self.fc2= nn.Linear(128, 3)
        self.dropout= nn.Dropout(p=0.4)
    def forward(self, x):
        x=x.to(torch.float32) 
        x= self.pool(torch.relu(self.conv1(x))) 
        x= self.pool(torch.relu(self.conv2(x))) 
        x= x.view(x.shape[0], x.shape[1]*x.shape[2])
        x= torch.relu(self.fc1(x))
        x= self.fc2(x)
        x= self.dropout(x)
        return x
    
class CNNModule(pl.LightningModule):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.model = CNN1D()
        self.loss = nn.BCELoss()
        self.gradient_clip_val= 1.0
        self.learning_rate=0.001
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

        predicted_classes = torch.argmax(y_hat, dim=1)
        targets = torch.argmax(y, dim=1)
        correct_predictions = (predicted_classes == targets).float()
        accuracy = correct_predictions.mean()
        self.log('test_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005)
        
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Learning rate scheduler
        return optimizer
