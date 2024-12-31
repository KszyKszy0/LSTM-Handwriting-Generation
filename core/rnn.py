import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_mixtures):
        super(MixtureDensityNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Output size = 6 * num_mixtures + 1:
        #   - 2 * num_mixtures (means for x and y)
        #   - 2 * num_mixtures (std deviations for x and y)
        #   - num_mixtures (correlation coefficients)
        #   - num_mixtures (mixture weights)
        #   - 1 (end-of-stroke probability)
        output_size = 6 * num_mixtures + 1
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Fully connected layer for outputs
        output = self.fc(lstm_out)

        # Extract different components
        end_of_stroke = torch.sigmoid(output[..., 0])  # End-of-stroke probability
        mixture_weights = F.softmax(output[..., 1:self.num_mixtures + 1], dim=-1)  # Mixture weights
        means = output[..., self.num_mixtures + 1:3 * self.num_mixtures + 1].view(-1, x.size(1), self.num_mixtures, 2)  # Means for x, y
        std_devs = torch.exp(output[..., 3 * self.num_mixtures + 1:5 * self.num_mixtures + 1].view(-1, x.size(1), self.num_mixtures, 2))  # Std deviations
        correlations = torch.tanh(output[..., 5 * self.num_mixtures + 1:6 * self.num_mixtures + 1])  # Correlation coefficients

        return end_of_stroke, mixture_weights, means, std_devs, correlations, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for LSTM
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Hyperparameters
# input_size = 3  # x, y offsets and binary end-of-stroke feature
# hidden_size = 400  # Hidden size of LSTM
# num_layers = 3  # Number of LSTM layers
# num_mixtures = 20  # Number of mixture components

# # Create the model
# model = MixtureDensityNetwork(input_size, hidden_size, num_layers, num_mixtures)

# # Example usage
# batch_size = 16
# sequence_length = 100
# x = torch.randn(batch_size, sequence_length, input_size)  # Random input sequence
# hidden = model.init_hidden(batch_size)

# # Forward pass
# end_of_stroke, mixture_weights, means, std_devs, correlations, hidden = model(x, hidden)

# print("End-of-stroke shape:", end_of_stroke.shape)
# print("Mixture weights shape:", mixture_weights.shape)
# print("Means shape:", means.shape)
# print("Std deviations shape:", std_devs.shape)
# print("Correlations shape:", correlations.shape)




def mdn_loss(end_of_stroke, mixture_weights, means, std_devs, correlations, target):
    """
    Funkcja liczy stratę MDN na podstawie parametrów wyjściowych modelu i danych docelowych.
    """
    # print('eos', end_of_stroke)
    # print('weights', mixture_weights)
    # print('means', means)
    # print('devs', std_devs)
    # print('correls', correlations)
    # print('target', target)

    # print('eos', end_of_stroke.shape)
    # print('weights', mixture_weights.shape)
    # print('means', means.shape)
    # print('devs', std_devs.shape)
    # print('correls', correlations.shape)
    # print('target', target.shape)

    x, y = target[..., 0], target[..., 1]
    eos_target = target[..., 2]

    # Dodanie osi do x i y, aby dopasować wymiary do means i std_devs
    x = x.unsqueeze(-1)  # Teraz x ma wymiar [1, 505, 1]
    y = y.unsqueeze(-1)  # Teraz y ma wymiar [1, 505, 1]

    # Parametry wyjściowe
    mean_x, mean_y = means[..., 0], means[..., 1]
    std_x, std_y = std_devs[..., 0], std_devs[..., 1]
    rho = correlations

    rho = torch.clamp(rho, -1 + 1e-5, 1 - 1e-5)

    # Obliczenie normalizowanego składnika Gaussowskiego
    z_x = ((x - mean_x) / std_x)
    z_y = ((y - mean_y) / std_y)
    z = (z_x ** 2) + (z_y ** 2) - 2 * rho * z_x * z_y
    norm_const = (1 - rho ** 2).sqrt()
    gaussian = (torch.exp(-z / (2 * norm_const ** 2)) /
                (2 * np.pi * std_x * std_y * norm_const))

    # Waga komponentów mieszanki
    mixture_prob = (mixture_weights * gaussian).sum(dim=-1)
    mixture_loss = -torch.log(mixture_prob + 1e-8).mean()

    # Koniec rysowania - strata binarna
    eos_loss = F.binary_cross_entropy(end_of_stroke, eos_target)

    return mixture_loss + eos_loss