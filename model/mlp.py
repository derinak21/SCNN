import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
#Define the MLP model 



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
            nn.Flatten(),
            nn.Linear(200, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16), 
            nn.ReLU(),  
            nn.Linear(16, 3),
            nn.Sigmoid()
        )   
    def forward(self, x):
        x=x.to(torch.float32)
        return self.layers(x)

#Define the lightning module using MLP model

class MLPModule(pl.LightningModule):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.model=MLP()
        self.loss=nn.CrossEntropyLoss()
          
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

        predicted_classes = torch.argmax(y_hat, dim=1)
        targets= torch.argmax(y, dim=1)

        correct_predictions = (predicted_classes==targets).float()
        accuracy = correct_predictions.mean()

        # accuracy= custom_accuracy(torch.round(prediction), y.float())
        self.log('test_accuracy', accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Learning rate scheduler

        return [optimizer], [scheduler]


