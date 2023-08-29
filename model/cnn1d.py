import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_

#Define the CNN model 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1= nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2= nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1= nn.Linear(32*125*250, 128)
        self.fc2= nn.Linear(128, 3)
        self.dropout= nn.Dropout(p=0.4)
    def forward(self, x):
        x=x.to(torch.float32)   # x.shape=[25(batch size), 500(height), 1000(width), 4(channels)]
        x= x.permute(0, 3, 1, 2).contiguous()
        x= self.pool(torch.relu(self.conv1(x))) # x.shape=[25, 16, 250, 500]
        x= self.pool(torch.relu(self.conv2(x))) # x.shape=[25, 32, 125, 250]
        x= x.view(x.size(0), -1)
        x= torch.relu(self.fc1(x))
        x= self.fc2(x)
        x= self.dropout(x)
        return x
    
class CNNModule(pl.LightningModule):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.model = CNN()
        self.loss = nn.CrossEntropyLoss()
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
        optimizer = optim.Adam(self.parameters(), lr=(self.learning_rate or self.lr))
        
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Learning rate scheduler
        return optimizer
