import torch
from torch.nn import Module

class MSELoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
    def forward(self, y_hat, y):
        return self.loss(y_hat, y)
    
class BCELoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCELoss()
    def forward(self, y_hat, y):
        return self.loss(y_hat, y)


    