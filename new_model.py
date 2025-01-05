import torch
import torch.nn as nn

class model(nn.Module):

    '''
        input will be alphabet size + 3 values x, y and probabilty 0-1 eos
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.f_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)
        self.eos = nn.Sigmoid()
        self.alphabetProbs = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.f_layer(x)
        x, hid = self.lstm(x)

        end_of_stroke = self.eos(x[:,:,2])

        # alphabet_probs = self.alphabetProbs(x[:,:,3:56])

        end_prob = self.eos(x[:,:,56])

        return x[:,:,:2], end_of_stroke, x[:,:,3:56], end_prob