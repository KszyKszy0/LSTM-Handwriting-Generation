import torch
import torch.nn as nn

class model(nn.Module):

    '''
        input will be alphabet size + 3 values x, y and probabilty 0-1 eos
    '''
    def __init__(self, input_size, hidden_size, output_size, layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = layers

        self.f_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.eos = nn.Sigmoid()
        self.alphabetProbs = nn.Softmax(dim=-1)


    def forward(self, x, hid=None):

        if hid is None:
            hid = self.init_hidden(x.size(0))

        x = self.f_layer(x)
        x, hid = self.lstm(x, hid)

        x = self.output_layer(x)

        end_of_stroke = self.eos(x[:,:,2])

        # alphabet_probs = self.alphabetProbs(x[:,:,3:56])

        end_prob = self.eos(x[:,:,56])

        return x[:,:,:2], end_of_stroke, x[:,:,3:56], end_prob, hid

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for LSTM
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))