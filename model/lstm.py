import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(512, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)
        self.sg= nn.Sigmoid()
        self.bn = nn.BatchNorm1d(4)  # Batch normalization after the linear layer
        self.apply(lambda m: setattr(m, "dtype", torch.float32))

    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])  # Take the last time step's output
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sg(out)
        out = self.bn(out)
        return out

class LSTMModule(pl.LightningModule):
    def __init__(self):
        super(LSTMModule, self).__init__()
        self.model = LSTM()
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        y_hat = self(x)
        y_hat= y_hat.sum(dim=1)
        loss = self.loss(y_hat, y.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat= y_hat.sum(dim=1)
        loss = self.loss(y_hat, y.float())
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat= y_hat.sum(dim=1)
        loss = self.loss(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)

        # predicted_classes = torch.argmax(y_hat, dim=1)
        # targets = torch.argmax(y, dim=1)
        # correct_predictions = (y == y_hat).float()
        # accuracy = correct_predictions.mean()
        mae = torch.abs(y_hat - y).mean()  # Calculate Mean Absolute Error (MAE)
        self.log('test_mae', mae, prog_bar=True)
        accuracy= (torch.round(y_hat)==y).float().mean()
        self.log('test_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Learning rate scheduler
        return [optimizer], [scheduler]


# import torch
# from torch import nn
# import torch.optim as optim
# import pytorch_lightning as pl


# class LSTM(nn.Module):
#     def __init__(self):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(512, 64, batch_first=True)
#         self.fc1 = nn.Linear(64, 16)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(16, 4)

#     def forward(self, x):
#         x=x.to(torch.float32)
#         #convert h0 to torch.float32
        
#         out, _ = self.lstm(x)
#         out = self.fc1(out[:, -1, :])  # Take the last time step's output
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out
    
# class LSTMModule(pl.LightningModule):
#     def __init__(self):
#         super(LSTMModule, self).__init__()
#         self.model=LSTM()
#         self.loss=nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch 
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         self.log('train_loss', loss, prog_bar=True)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         self.log('val_loss', loss, prog_bar=True)

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         self.log('test_loss', loss, prog_bar=True)

#         predicted_classes = torch.argmax(y_hat, dim=1)
#         targets= torch.argmax(y, dim=1)
#         correct_predictions = (predicted_classes == targets).float()
#         accuracy = correct_predictions.mean()
#         self.log('test_accuracy', accuracy, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=0.05)
#         return optimizer