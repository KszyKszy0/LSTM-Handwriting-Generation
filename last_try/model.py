import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================
# 1. Define Character Dictionary
# =========================
# Here we define a simple vocabulary. You can expand it as needed.
vocab = list("abcdefghijklmnopqrstuvwxyz 1234567890")  # letters and space
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}

# =========================
# 2. Handwriting RNN with Window and MDN Output
# =========================
class HandwritingRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_mixtures, char_vocab_size, window_mixtures):
        """
        Args:
          input_dim: Dimensionality of the pen stroke input (here, 3: x, y, end-of-stroke)
          hidden_dim: Hidden state dimension for the LSTM.
          num_layers: Number of LSTM layers.
          num_mixtures: Number of Gaussian mixtures for the MDN output.
          char_vocab_size: Size of the character vocabulary.
          window_mixtures: Number of Gaussian components for the window (attention) mechanism.
        """
        super(HandwritingRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.char_vocab_size = char_vocab_size
        self.window_mixtures = window_mixtures

        # LSTM will take a concatenated input of (pen stroke + window vector)
        # So the input size to the LSTM is input_dim + char_vocab_size.
        self.lstm = nn.LSTM(input_dim + char_vocab_size, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer for MDN outputs.
        # For each mixture, we need to output:
        #   - mixture weight (pi)
        #   - μ₁, μ₂ (means for x and y)
        #   - σ₁, σ₂ (standard deviations for x and y)
        #   - ρ (correlation coefficient)
        # Plus one extra output for the end-of-stroke probability.
        # That gives a total output dimension of 6*num_mixtures + 1.
        self.fc_mdn = nn.Linear(hidden_dim, 6 * num_mixtures + 1)

        # Fully connected layer for window parameters.
        # For each window mixture we predict:
        #   - delta_kappa (the increment for kappa)
        #   - α (weight)
        #   - β (scale parameter)
        # Total = 3 * window_mixtures.
        self.fc_window = nn.Linear(hidden_dim, 3 * window_mixtures)

        # We will set the character dictionary externally (see below in usage)
        self.char_to_idx = None

    def forward(self, input_seq, text):
        """
        Forward pass for a sequence.
        
        Args:
          input_seq: Tensor of shape (batch, seq_len, 3) with pen strokes.
          text: List of strings (length=batch) that represent the text (word) to condition on.
        
        Returns:
          mdn_params_seq: Tensor of shape (batch, seq_len, 6*num_mixtures + 1) containing the MDN parameters for each time step.
        """
        batch_size, seq_len, _ = input_seq.size()
        
        # -------------------------
        # Encode the text as one-hot vectors.
        # We get a tensor of shape (batch, max_text_len, char_vocab_size).
        # Also retrieve lengths for each text (if needed for masking).
        # -------------------------
        text_encoded, text_lengths = self.encode_text_batch(text)  # see helper method below
        max_text_len = text_encoded.size(1)  # maximum text length in the batch

        # -------------------------
        # Initialize the window mechanism.
        # prev_kappa holds the cumulative positions for each window mixture.
        # window_vec is the computed window vector (weighted sum over the text one-hot vectors).
        # -------------------------
        prev_kappa = torch.zeros(batch_size, self.window_mixtures, device=input_seq.device)
        window_vec = torch.zeros(batch_size, self.char_vocab_size, device=input_seq.device)

        # This list will collect the MDN output parameters at each time step.
        outputs = []

        # Hidden state for the LSTM. Passing None means the LSTM will use a zero-initialized state.
        hidden = None

        # Process each time step sequentially.
        for t in range(seq_len):
            # -------------------------
            # Get the current pen stroke input: shape (batch, 3)
            # -------------------------
            x_t = input_seq[:, t, :]  # current stroke (x, y, end-of-stroke indicator)

            # -------------------------
            # Concatenate the current pen stroke with the previous window vector.
            # This forms the input to the LSTM at the current time step.
            # -------------------------
            lstm_input = torch.cat([x_t, window_vec], dim=1).unsqueeze(1)  # shape becomes (batch, 1, input_dim + char_vocab_size)

            # -------------------------
            # Process one LSTM step.
            # out: output for the current time step, hidden: updated hidden state.
            # -------------------------
            out, hidden = self.lstm(lstm_input, hidden)
            h_t = out.squeeze(1)  # shape (batch, hidden_dim)

            # -------------------------
            # Compute window parameters from the hidden state.
            # We use a fully connected layer to predict parameters for each window mixture.
            # -------------------------
            window_params = self.fc_window(h_t)  # shape: (batch, 3*window_mixtures)
            # Reshape to separate out the window mixtures and their 3 parameters.
            window_params = window_params.view(batch_size, self.window_mixtures, 3)
            # Split into delta_kappa, α (alpha), and β (beta).
            delta_kappa = window_params[:, :, 0]  # (batch, window_mixtures)
            alpha = window_params[:, :, 1]          # (batch, window_mixtures)
            beta = window_params[:, :, 2]           # (batch, window_mixtures)

            # -------------------------
            # Apply activations to ensure positivity.
            # We use exp() so that delta_kappa, alpha, and beta are > 0.
            # -------------------------
            delta_kappa = torch.exp(delta_kappa)
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)

            # -------------------------
            # Update kappa: the position parameter in the window.
            # The new kappa is the previous kappa plus delta_kappa.
            # This ensures a monotonic (increasing) progression over the text.
            # -------------------------
            kappa = prev_kappa + delta_kappa  # shape (batch, window_mixtures)
            prev_kappa = kappa  # update for the next time step

            # -------------------------
            # Compute the window (attention over text).
            # For each position u in the text, compute a weighted contribution.
            # The weight is given by a mixture of Gaussians:
            #   φ_t(u) = sum_j [α_j * exp(-β_j * (kappa_j - u)^2)]
            # -------------------------
            # Create a tensor u representing the positions in the text: shape (max_text_len,)
            u = torch.arange(0, max_text_len, device=input_seq.device).float()  # positions 0, 1, ..., max_text_len-1
            # Reshape u for broadcasting: shape (1, 1, max_text_len)
            u = u.view(1, 1, -1)
            # Expand kappa and beta to compute the Gaussian: shape (batch, window_mixtures, 1)
            kappa_expanded = kappa.unsqueeze(2)
            beta_expanded = beta.unsqueeze(2)
            # Compute φ for each window mixture and each text position.
            # Result: (batch, window_mixtures, max_text_len)
            phi = alpha.unsqueeze(2) * torch.exp(-beta_expanded * (kappa_expanded - u) ** 2)
            # Sum over the window mixtures to get a weight for each text position.
            # φ now has shape (batch, max_text_len)
            phi = phi.sum(dim=1)

            # -------------------------
            # Compute the window vector.
            # Multiply the weights φ with the one-hot text encoding and sum over the text positions.
            # text_encoded is (batch, max_text_len, char_vocab_size), so we do a weighted sum over dimension 1.
            # -------------------------
            # First, unsqueeze φ to shape (batch, 1, max_text_len) so we can use batch matrix multiplication.
            phi_unsqueezed = phi.unsqueeze(1)
            # Perform the weighted sum: result shape is (batch, 1, char_vocab_size)
            window_vec = torch.bmm(phi_unsqueezed, text_encoded)
            window_vec = window_vec.squeeze(1)  # shape (batch, char_vocab_size)

            # -------------------------
            # Compute the MDN output parameters from the hidden state.
            # This layer predicts all the parameters needed to form the mixture of Gaussians.
            # -------------------------
            mdn_params = self.fc_mdn(h_t)  # shape (batch, 6*num_mixtures + 1)
            outputs.append(mdn_params)

        # Stack the outputs from all time steps.
        # Final shape: (batch, seq_len, 6*num_mixtures + 1)
        mdn_params_seq = torch.stack(outputs, dim=1)
        return mdn_params_seq
    
    def generate_step(self, x_t, hidden, prev_kappa, window_vec, text_encoded):
        """
        Perform one generation step.
        
        Args:
          x_t: Current pen stroke input of shape (batch, input_dim).
          hidden: Previous LSTM hidden state.
          prev_kappa: Previous window position parameters, shape (batch, window_mixtures).
          window_vec: Previous window vector of shape (batch, char_vocab_size).
          text_encoded: One-hot encoded text of shape (batch, max_text_len, char_vocab_size).
          
        Returns:
          mdn_params: MDN raw output for the current step, shape (batch, 6*num_mixtures+1).
          hidden: Updated LSTM hidden state.
          kappa: Updated window position (to be used in next step).
          window_vec: Updated window vector.
        """
        batch_size = x_t.size(0)  # typically 1 during generation
        # Concatenate current stroke and window vector.
        lstm_input = torch.cat([x_t, window_vec], dim=1).unsqueeze(1)
        out, hidden = self.lstm(lstm_input, hidden)
        h_t = out.squeeze(1)
        # Compute window parameters.
        window_params = self.fc_window(h_t).view(batch_size, self.window_mixtures, 3)
        delta_kappa = torch.exp(window_params[:, :, 0])
        alpha = torch.exp(window_params[:, :, 1])
        beta = torch.exp(window_params[:, :, 2])
        kappa = prev_kappa + delta_kappa  # monotonic increase
        # Compute the attention (window) over the text.
        max_text_len = text_encoded.size(1)
        u = torch.arange(0, max_text_len, device=x_t.device).float().view(1, 1, -1)
        phi = alpha.unsqueeze(2) * torch.exp(-beta.unsqueeze(2) * (kappa.unsqueeze(2) - u)**2)
        phi = phi.sum(dim=1)
        phi_unsqueezed = phi.unsqueeze(1)
        window_vec = torch.bmm(phi_unsqueezed, text_encoded).squeeze(1)
        # Compute MDN parameters.
        mdn_params = self.fc_mdn(h_t)
        return mdn_params, hidden, kappa, window_vec

    def encode_text_batch(self, text_batch):
        """
        Convert a list of strings into a one-hot encoded tensor.
        
        Args:
          text_batch: List of strings (length=batch).
          
        Returns:
          text_encoded: Tensor of shape (batch, max_text_length, char_vocab_size) with one-hot vectors.
          text_lengths: List containing the length of each text string.
        """
        batch_size = len(text_batch)
        # Determine lengths and maximum text length in the batch.
        text_lengths = [len(t) for t in text_batch]
        max_len = max(text_lengths)

        # Initialize a tensor of zeros for one-hot encoding.
        # We assume self.char_vocab_size is already set.
        text_tensor = torch.zeros(batch_size, max_len, self.char_vocab_size, device=torch.device("cpu"))
        for i, text in enumerate(text_batch):
            for j, char in enumerate(text.lower()):
                # Use the character dictionary provided (if a char is not found, it is skipped)
                idx = self.char_to_idx.get(char, None)
                if idx is not None:
                    text_tensor[i, j, idx] = 1
        return text_tensor, text_lengths
    



'''LOSS FUNCTION'''




# Assume that `model`, `dataloader`, and `num_mixtures` are defined as in the previous code snippet.
# For example, model is an instance of HandwritingRNN with MDN output of shape (batch, seq_len, 6*num_mixtures+1)
# and dataloader yields (input_seq, target_seq, text).

def mdn_loss(mdn_params_seq, target_seq, num_mixtures):
    """
    Computes the loss for the MDN output plus pen state for handwriting synthesis.
    
    Args:
      mdn_params_seq: Tensor of shape (B, T, 6*num_mixtures+1), raw outputs from the network.
      target_seq: Tensor of shape (B, T, 3) with the target (x, y, pen state).
      num_mixtures: Number of Gaussian mixtures (M) used in the MDN.
    
    Returns:
      loss: A scalar tensor representing the average loss over the batch.
    """
    B, T, _ = mdn_params_seq.shape
    M = num_mixtures

    # -------------------------
    # Split the MDN output into its components.
    # The ordering is assumed to be:
    #   - pi (mixture coefficients): first M values
    #   - mu1: next M values
    #   - mu2: next M values
    #   - sigma1: next M values
    #   - sigma2: next M values
    #   - rho: next M values
    #   - pen_logits: last value (for pen state probability)
    # -------------------------
    pi = mdn_params_seq[:, :, :M]                           # (B, T, M)
    mu1 = mdn_params_seq[:, :, M:2*M]                         # (B, T, M)
    mu2 = mdn_params_seq[:, :, 2*M:3*M]                       # (B, T, M)
    sigma1 = mdn_params_seq[:, :, 3*M:4*M]                    # (B, T, M)
    sigma2 = mdn_params_seq[:, :, 4*M:5*M]                    # (B, T, M)
    rho = mdn_params_seq[:, :, 5*M:6*M]                       # (B, T, M)
    pen_logits = mdn_params_seq[:, :, -1]                     # (B, T)

    # -------------------------
    # Apply activation functions:
    # - Softmax on pi to make them probabilities.
    # - Exponential on sigma1 and sigma2 to ensure they are positive.
    # - Tanh on rho to keep it in [-1, 1].
    # - Sigmoid on pen_logits to get pen state probability.
    # -------------------------
    pi = F.softmax(pi, dim=-1)
    sigma1 = torch.exp(sigma1)
    sigma2 = torch.exp(sigma2)
    rho = torch.tanh(rho)
    pen_prob = torch.sigmoid(pen_logits)

    # -------------------------
    # Extract target values.
    # target_seq is assumed to have 3 channels: x, y, and pen state.
    # -------------------------
    x_target = target_seq[:, :, 0]    # (B, T)
    y_target = target_seq[:, :, 1]    # (B, T)
    pen_target = target_seq[:, :, 2]  # (B, T), assumed to be 0 or 1

    # -------------------------
    # Compute the probability density of the target (x, y) under each Gaussian mixture component.
    # We first expand the target dimensions to align with the mixture components.
    # -------------------------
    x_target_exp = x_target.unsqueeze(-1)  # (B, T, 1)
    y_target_exp = y_target.unsqueeze(-1)  # (B, T, 1)

    # Compute normalized differences.
    norm_x = (x_target_exp - mu1) / sigma1
    norm_y = (y_target_exp - mu2) / sigma2

    # Compute the exponent term for the bivariate Gaussian.
    # Note: The bivariate normal density is defined as:
    #   N(x,y) = 1/(2πσ1σ2√(1-ρ²)) * exp{ -1/(2(1-ρ²)) * [norm_x² + norm_y² - 2ρ norm_x norm_y] }
    z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    denom = 2 * (1 - rho**2) + 1e-8  # epsilon for stability
    exponent = -z / denom

    # Normalizing constant for each component.
    normalizer = 2 * math.pi * sigma1 * sigma2 * torch.sqrt(1 - rho**2 + 1e-8)
    
    # Probability for each component.
    component_prob = torch.exp(exponent) / (normalizer + 1e-8)  # (B, T, M)

    # -------------------------
    # Weight by the mixture coefficients and sum across mixtures.
    # This gives the total probability density for the target (x, y) at each time step.
    # -------------------------
    weighted_prob = pi * component_prob  # (B, T, M)
    prob = torch.sum(weighted_prob, dim=-1) + 1e-8  # (B, T); add epsilon to avoid log(0)

    # Negative log-likelihood for the coordinates.
    loss_mdn = -torch.log(prob)  # (B, T)

    # -------------------------
    # Pen state loss:
    # Use binary cross-entropy between the predicted pen probability and target.
    # -------------------------
    loss_pen = F.binary_cross_entropy(pen_prob, pen_target, reduction='none')  # (B, T)

    # Total loss at each time step is the sum of the MDN loss and the pen loss.
    loss = loss_mdn + loss_pen

    # Average loss over all time steps and the batch.
    return torch.mean(loss)