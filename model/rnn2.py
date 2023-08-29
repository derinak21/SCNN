import torch 
import torch.nn as nn

#RNN binary classifier

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)
        #convert h0 to torch.float32
        h0 = torch.zeros(1, 64)
        h0 = h0.to(torch.float32)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out)

#Define RNN classifier with LSTM

class RNNClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(torch.float32)
        #convert h0 to torch.float32
        h0 = torch.zeros(1, 64)
        h0 = h0.to(torch.float32)
        c0 = torch.zeros(1, 64)
        c0 = c0.to(torch.float32)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out= self.sigmoid(out)
        return out