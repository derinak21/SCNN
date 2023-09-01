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
        # self.loss=FocalLoss(alpha=0.5, gamma=2, reduction='mean')
        # self.loss=nn.HingeEmbeddingLoss()
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

        correct_predictions = (predicted_classes==targets).float()
        accuracy = correct_predictions.mean()

        # accuracy= custom_accuracy(torch.round(prediction), y.float())
        self.log('test_accuracy', accuracy, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005, weight_decay=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler

        return [optimizer], [scheduler]


