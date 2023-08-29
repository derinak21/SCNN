import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

def custom_accuracy(predictions, targets):
    differences = torch.abs(predictions - targets)
    max_difference = torch.max(differences)
    normalized_differences = differences / max_difference
    accuracy = 1.0 - normalized_differences
    accuracy = torch.clamp(accuracy, min=0.0, max=1.0)
    mean_accuracy = torch.mean(accuracy)
    return mean_accuracy



class BinaryClassifierMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifierMLP, self).__init__()
        self.layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, 64), 
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16), 
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )  
          
    def forward(self, x):
        x=x.to(torch.float32)
        return self.layers(x)

class MultiClassifierMLP(nn.Module):
    def __init__(self, num_iterations, input_size, hidden_size):
        super(MultiClassifierMLP, self).__init__()
        self.iteration_modules= nn.ModuleList([BinaryClassifierMLP(input_size, hidden_size) for i in range(num_iterations)])
        self.num_iterations = num_iterations

    def forward(self, x, current_iteration):
        module= self.iteration_modules[current_iteration]
        return module(x)



class MultiClassifictionModule(pl.LightningModule):
    def __init__(self, num_iterations, input_size, hidden_size):
        super(MultiClassifictionModule, self).__init__()
        self.model=MultiClassifierMLP(num_iterations, input_size, hidden_size)
        self.loss=nn.BCELoss()
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.05)
        return optimizer
    
    def forward(self, x, current_iteration):
        return self.model(x, current_iteration)
    
    def training_step(self, batch, batch_idx):
        x, y_one_hot = batch 
        total_loss = 0
        optimizer = self.optimizers()
        mask = torch.zeros(len(batch[0]), 1).float()   
        for current_iteration in range(4):
            output = self(x, current_iteration)
            masked_output = output * (1 - mask) 
            y= y_one_hot[:, current_iteration].view(-1, 1)
            mask = ((output == 0) | (output >= 0.5)).float()
            loss = self.loss(masked_output, y)
            total_loss += loss * (current_iteration+1) * 0.1
            self.log('batch_loss', loss, prog_bar=True) 
            self.manual_backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        self.log('train_loss', total_loss, prog_bar=True) 
        return total_loss
        
    
    def validation_step(self, batch, batch_idx):
        x, y_one_hot = batch 
        total_loss = 0
        mask = torch.zeros(len(batch[0]), 1).float()   
        for current_iteration in range(4):
            output = self(x, current_iteration)
            masked_output = output * (1 - mask) 
            y= y_one_hot[:, current_iteration].view(-1, 1)
            mask = ((output == 0) | (output >= 0.5)).float()
            loss = self.loss(masked_output, y)
            total_loss += loss * (current_iteration+1) * 0.1
            self.log('batch_loss', loss, prog_bar=True) 
        self.log('val_loss', total_loss, prog_bar=True) 
       

    def test_step(self, batch, batch_idx):
        x, y_one_hot = batch 
        total_loss = 0
        mask = torch.zeros(len(batch[0]), 1).float()
        outputs=[]
        for current_iteration in range(4):
            output = self(x, current_iteration)
            masked_output = output * (1 - mask) 
            y= y_one_hot[:, current_iteration].view(-1, 1)
            mask = ((output == 0) | (output >= 0.5)).float()
            loss = self.loss(masked_output, y)
            total_loss += loss * (current_iteration+1) * 0.1
            self.log('test_batch_loss', loss, prog_bar=True) 
            outputs.append(masked_output)
        matrix=torch.stack(outputs)
        predictions=torch.max(matrix, dim=0)

        accuracy= (torch.max(y_one_hot, dim=1).indices==predictions.indices).float().mean()
        self.log('test_loss', total_loss, prog_bar=True) 
        self.log('test_accuracy', accuracy, prog_bar=True)
        
  

