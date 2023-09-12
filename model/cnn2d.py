import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        
#Define the CNN model 
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1= nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1= nn.Linear(90112, 128)
        self.fc2= nn.Linear(128, 3)
        self.dropout= nn.Dropout(p=0.4)
        self.softmax= nn.Softmax(dim=1)
    def forward(self, x):
        x=x.to(torch.float32)   # x.shape=[25(batch size), 500(height), 1000(width), 4(channels)]
        x= x.view(x.size(0), 1, x.size(1), x.size(2))
        x= self.pool(torch.relu(self.conv1(x))) # x.shape=[25, 16, 250, 500]
        x= self.pool(torch.relu(self.conv2(x))) # x.shape=[25, 32, 125, 250]
        x= x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x= torch.relu(self.fc1(x))
        x= self.fc2(x)
        x= self.dropout(x)
        # x= self.softmax(x)
        return x
    
class CNN2DModule(pl.LightningModule):
    def __init__(self, loss, weight_decay, learning_rate, scheduler):
        super(CNN2DModule, self).__init__()
        self.model = CNN2D()
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

   