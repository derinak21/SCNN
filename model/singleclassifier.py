import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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


# class Classifier2(nn.Module):
#     def __init__(self):
#         super(Classifier2, self).__init__()
#         self.layers=nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(201, 64), 
#             nn.ReLU(),
#             nn.Linear(64, 32), 
#             nn.ReLU(), 
#             nn.Linear(32, 2), 
#             nn.ReLU(),  
#             nn.Linear(2, 1),
#             nn.ReLU()
#         )   
#         self.final_layer=nn.Linear(2, 1)

#     def forward(self, x, prev_prediction):
#         x=x.to(torch.float32)
#         x_with_prev_prediction= torch.cat((x, (prev_prediction)), dim=1)
#         final= self.layers(x_with_prev_prediction)
#         x_f=torch.cat((final, prev_prediction), dim=1)
#         final_2= self.final_layer(x_f)
#         return torch.sigmoid(final_2)
    

class ClassifierModule(pl.LightningModule):
    def __init__(self, iteration):
        super(ClassifierModule, self).__init__()
        self.iteration=iteration
        self.model=Classifier()

        # if self.iteration==0:
        # else:
        #     self.model=Classifier2()
        self.loss=nn.BCELoss()
        # self.train_predictions=[]
        # self.val_predictions=[]
        self.test_predictions=[]
        # self.train_prev_prediction=train_prev_prediction
        # self.val_prev_prediction=val_prev_prediction
        # self.test_prev_prediction=test_prev_prediction

    def forward(self, x, prev_prediction=None):
        # if self.iteration==0:
        return self.model(x)
        # else: 
        #     return self.model(x, prev_prediction)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        # if self.iteration==0:
        y_hat = self(x)
        # else:
        #     prev_prediction = self.train_prev_prediction[batch_idx].detach().clone()
        #     y_hat = self(x, prev_prediction)
        y= (y>(self.iteration)).view(-1,1).float()
        loss = self.loss(y_hat, y.float())
        self.log('train_loss', loss, prog_bar=True)
        # self.train_predictions.append(torch.round(y_hat))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # if self.iteration==0:
        y_hat = self(x)
        # else:
        #     prev_prediction = self.val_prev_prediction[batch_idx].detach().clone()
        #     y_hat = self(x, prev_prediction)        
        y= (y>(self.iteration)).view(-1,1).float()
        loss = self.loss(y_hat, y.float())
        self.log('val_loss', loss, prog_bar=True)
        # self.val_predictions.append(torch.round(y_hat))

    def test_step(self, batch, batch_idx):
        x, y = batch
        # if self.iteration==0:
        y_hat = self(x)
        # else:
        #     prev_prediction = self.test_prev_prediction[batch_idx].detach().clone()
        #     y_hat = self(x, prev_prediction)
        y= (y>(self.iteration)).view(-1,1).float()
        loss = self.loss(y_hat, y.float())
        self.log('test_loss', loss, prog_bar=True)
        accuracy = torch.mean((torch.round(y_hat) == y).float())
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.test_predictions.append(torch.round(y_hat))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=(0.05-0.01*self.iteration))
        return optimizer
