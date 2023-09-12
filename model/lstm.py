import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(512, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)
        self.st= nn.Softmax(dim=1)
        self.apply(lambda m: setattr(m, "dtype", torch.float32))

    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.lstm(x)
        out = self.fc1(torch.mean(out, dim=1))   # Take the last time step's output: out = self.fc1(out[:, -1, :]) 
        out = self.relu(out)
        out = self.fc2(out)
        out = self.st(out)
        return out
    
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
class LSTMModule(pl.LightningModule):
    def __init__(self, loss, weight_decay, learning_rate, scheduler):
        super(LSTMModule, self).__init__()
        self.model = LSTM()
        lossdict={
            'BCELoss': nn.BCELoss(),
            'MSELoss': nn.MSELoss(),
            'CrossEntropyLoss': nn.CrossEntropyLoss(),
            'FocalLoss': FocalLoss(alpha=0.5, gamma=2, reduction='mean'),
        }
        self.loss=lossdict[loss]
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.scheduler=scheduler

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
        predicted_classes = torch.argmax(y_hat, dim=1)
        targets = torch.argmax(y, dim=1)
        correct_predictions = (predicted_classes == targets).float()
        accuracy = correct_predictions.mean()
        
        self.log('val_accuracy', accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)

        predicted_classes = torch.argmax(y_hat, dim=1)
        targets = torch.argmax(y, dim=1)
        correct_predictions = (predicted_classes == targets).float()
        accuracy = correct_predictions.mean()
        self.log('test_accuracy', accuracy, prog_bar=True)
        precision = precision_score(targets.cpu(), predicted_classes.cpu(), average='macro')
        recall = recall_score(targets.cpu(), predicted_classes.cpu(), average='macro')
        f1 = f1_score(targets.cpu(), predicted_classes.cpu(), average='macro')
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.scheduler is None:
            return optimizer
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Learning rate scheduler
            return [optimizer], [scheduler]



