import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixtureDensityNetworkWithWindow(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_mixtures, window_size, num_window_components=10):
        """
        Args:
            input_size: Dimensionality of the stroke input.
            hidden_size: Number of hidden units in the LSTM.
            num_layers: Number of LSTM layers.
            num_mixtures: Number of mixture components for the MDN.
            window_size: Dimensionality of the text embedding (the “window” vector).
            num_window_components: Number of Gaussian components to parameterize the attention window.
        """
        super(MixtureDensityNetworkWithWindow, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.window_size = window_size
        self.num_window_components = num_window_components

        # The LSTM now takes concatenated input: the original stroke input and the window vector.
        self.lstm = nn.LSTM(input_size + window_size, hidden_size, num_layers, batch_first=True)

        # Window layer: from the previous hidden state, produce parameters for the attention window.
        # For each of num_window_components, we output 3 parameters: alpha, beta, and delta_kappa.
        # In practice, we use an exponential to ensure α, β, and Δκ are positive.
        self.window_fc = nn.Linear(hidden_size, 3 * num_window_components)

        # MDN output layer. The output size remains 6 * num_mixtures + 1:
        #  - 2*num_mixtures: means for x and y
        #  - 2*num_mixtures: std deviations for x and y (after applying exp)
        #  - num_mixtures: correlation coefficients (after tanh)
        #  - num_mixtures: mixture weights (after softmax)
        #  - 1: end-of-stroke probability (after sigmoid)
        output_size = 6 * num_mixtures + 1
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, text, kappa_prev):
        """
        Args:
            x: Input stroke sequence of shape (batch, seq_len, input_size).
            hidden: Tuple of LSTM hidden and cell states.
            text: Text embeddings of shape (batch, text_len, window_size). Each row corresponds to a character (or its embedding).
            kappa_prev: Previous attention positions, shape (batch, num_window_components).

        Returns:
            A tuple containing:
              - end_of_stroke: (batch, seq_len) end-of-stroke probabilities.
              - mixture_weights: (batch, seq_len, num_mixtures) mixture weights.
              - means: (batch, seq_len, num_mixtures, 2) means for x and y.
              - std_devs: (batch, seq_len, num_mixtures, 2) standard deviations.
              - correlations: (batch, seq_len, num_mixtures) correlation coefficients.
              - hidden: Updated LSTM hidden state.
              - kappa: Updated attention positions to be fed in the next call.
        """
        batch_size, seq_len, _ = x.size()
        # print(text)
        # text_len = text.size(1)
        print(len(x),len(text))
        # text_len = 1

        outputs = []  # to collect LSTM outputs at each time step
        kappa = kappa_prev  # cumulative attention location, updated at each time step

        # Unroll the LSTM for each time step (to update the window each step)
        for t in range(seq_len):
            # For t==0, no previous hidden state is available for the window so use a zero vector.
            if t == 0:
                window_t = torch.zeros(batch_size, self.window_size, device=x.device)
            else:
                # Use the previous LSTM output to compute the window parameters.
                h_prev = outputs[-1]  # shape: (batch, hidden_size)
                window_params = self.window_fc(h_prev)  # shape: (batch, 3*num_window_components)
                window_params = window_params.view(batch_size, self.num_window_components, 3)
                # Split into parameters and ensure positivity via exp.
                alpha = torch.exp(window_params[:, :, 0])       # (batch, num_window_components)
                beta  = torch.exp(window_params[:, :, 1])        # (batch, num_window_components)
                delta_kappa = torch.exp(window_params[:, :, 2])    # (batch, num_window_components)
                # Update kappa: new kappa = previous kappa + delta_kappa.
                kappa = kappa + delta_kappa  # (batch, num_window_components)
                # Compute attention weights φ over the text positions.
                u = torch.arange(0, text_len, device=x.device).float()  # (text_len,)
                u = u.view(1, 1, -1)  # shape: (1, 1, text_len)
                kappa_exp = kappa.unsqueeze(-1)  # (batch, num_window_components, 1)
                beta_exp = beta.unsqueeze(-1)    # (batch, num_window_components, 1)
                # Compute Gaussian weights for each window component over text positions.
                phi_components = alpha.unsqueeze(-1) * torch.exp(-beta_exp * (kappa_exp - u) ** 2)
                # Sum over window components to get the final attention weight for each text position.
                phi = phi_components.sum(dim=1)  # (batch, text_len)
                # Compute the window vector as a weighted sum over the text embeddings.
                phi = phi.unsqueeze(1)  # (batch, 1, text_len)
                window_t = torch.bmm(phi, text)  # (batch, 1, window_size)
                window_t = window_t.squeeze(1)    # (batch, window_size)

            # Concatenate the stroke input at time t with the computed window vector.
            x_t = x[:, t, :]  # (batch, input_size)
            lstm_input = torch.cat([x_t, window_t], dim=-1).unsqueeze(1)  # (batch, 1, input_size + window_size)
            # Forward pass one time step through the LSTM.
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: (batch, 1, hidden_size)
            h_t = lstm_out.squeeze(1)  # (batch, hidden_size)
            outputs.append(h_t)

        # Stack the outputs from each time step into a single tensor.
        outputs = torch.stack(outputs, dim=1)  # shape: (batch, seq_len, hidden_size)

        # Compute the MDN outputs from the LSTM outputs.
        mdn_output = self.fc(outputs)  # (batch, seq_len, 6*num_mixtures + 1)

        # Parse the MDN outputs:
        end_of_stroke = torch.sigmoid(mdn_output[..., 0])
        mixture_weights = F.softmax(mdn_output[..., 1:self.num_mixtures+1], dim=-1)
        means = mdn_output[..., self.num_mixtures+1:3*self.num_mixtures+1]\
                    .view(batch_size, seq_len, self.num_mixtures, 2)
        std_devs = torch.exp(mdn_output[..., 3*self.num_mixtures+1:5*self.num_mixtures+1]\
                             .view(batch_size, seq_len, self.num_mixtures, 2))
        correlations = torch.tanh(mdn_output[..., 5*self.num_mixtures+1:6*self.num_mixtures+1])

        return end_of_stroke, mixture_weights, means, std_devs, correlations, hidden, kappa

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state for the LSTM.
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