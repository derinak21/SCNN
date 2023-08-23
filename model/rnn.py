import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn import precision_score, recall_score, f1_score
# Define the RNN model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=200, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = x.to(torch.float32)
        #convert h0 to torch.float32
        h0 = torch.zeros(1, 64)
        h0 = h0.to(torch.float32)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out)  # Take the last time step's output
        out = self.relu(out)
        out = self.fc2(out)
        out= self.softmax(out)
        return out


class RNNModule(pl.LightningModule):
    def __init__(self):
        super(RNNModule, self).__init__()
        self.model=RNN()
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

        precision= precision_score(targets, predicted_classes, average='weighted')
        recall = recall_score(targets, predicted_classes, average='weighted')
        f1 = f1_score(targets, predicted_classes, average='weighted')

        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)
        
        correct_predictions = (predicted_classes == targets).float()
        accuracy = correct_predictions.mean()
        self.log('test_accuracy', accuracy, prog_bar=True)



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer


