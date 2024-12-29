import torch
import torch.nn as nn
import torch.nn.functional as F

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
input_size = 3  # x, y offsets and binary end-of-stroke feature
hidden_size = 400  # Hidden size of LSTM
num_layers = 3  # Number of LSTM layers
num_mixtures = 20  # Number of mixture components

# Create the model
model = MixtureDensityNetwork(input_size, hidden_size, num_layers, num_mixtures)

# Example usage
batch_size = 16
sequence_length = 100
x = torch.randn(batch_size, sequence_length, input_size)  # Random input sequence
hidden = model.init_hidden(batch_size)

# Forward pass
end_of_stroke, mixture_weights, means, std_devs, correlations, hidden = model(x, hidden)

print("End-of-stroke shape:", end_of_stroke.shape)
print("Mixture weights shape:", mixture_weights.shape)
print("Means shape:", means.shape)
print("Std deviations shape:", std_devs.shape)
print("Correlations shape:", correlations.shape)