import torch
import torch.nn as nn

class model(nn.Module):

    '''
        input will be alphabet size + 3 values x, y and probabilty 0-1 eos
    '''
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.f_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, input_size)
        self.eos = nn.Sigmoid()

    def forward(self, x):
        x = self.f_layer(x)
        x = self.lstm(x)

        x[:,2] = self.eos(x[:,2])

        return x