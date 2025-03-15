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
    def __init__(self, input_dim, hidden_dim, num_mixtures, char_vocab_size, window_mixtures):
        """
        Args:
          input_dim: Dimensionality of the pen stroke input (3: x, y, pen state)
          hidden_dim: Hidden state dimension for both LSTM layers.
          num_mixtures: Number of Gaussian mixtures for the MDN output.
          char_vocab_size: Number of characters in the vocabulary.
          window_mixtures: Number of Gaussian components for the window (attention) mechanism.
        """
        super(HandwritingRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.char_vocab_size = char_vocab_size
        self.window_mixtures = window_mixtures

        # LSTM1: Processes the raw input (pen stroke + previous window vector)
        # We use an LSTMCell to have fine-grained control over the recurrence.
        self.lstm1 = nn.LSTMCell(input_dim + char_vocab_size, hidden_dim)
        
        # Window mechanism: from LSTM1 output we predict parameters for a mixture of Gaussians.
        # Each component produces: delta_kappa, alpha, beta.
        self.fc_window = nn.Linear(hidden_dim, 3 * window_mixtures)

        # LSTM2: Processes the concatenation of LSTM1's output and the updated window vector.
        self.lstm2 = nn.LSTMCell(hidden_dim + char_vocab_size + input_dim, hidden_dim)

        # MDN output layer: For each mixture component predict:
        #   pi, mu1, mu2, sigma1, sigma2, rho  (6 parameters per mixture)
        # plus one extra value for the pen (end-of-stroke) probability.
        self.fc_mdn = nn.Linear(hidden_dim, 6 * num_mixtures + 1)

        # The character dictionary will be set externally.
        self.char_to_idx = None

    def forward(self, input_seq, text):
        """
        Training forward pass over an entire sequence.
        
        Args:
          input_seq: Tensor of shape (batch, seq_len, 3) with pen strokes.
          text: List of strings (length=batch) representing the target text.
          
        Returns:
          mdn_params_seq: Tensor of shape (batch, seq_len, 6*num_mixtures+1).
        """
        batch_size, seq_len, _ = input_seq.size()
        device = input_seq.device
        
        # Encode text into one-hot vectors.
        text_encoded, text_lengths = self.encode_text_batch(text)
        max_text_len = text_encoded.size(1)
        text_encoded = text_encoded.to(device)

        # Initialize hidden states for both LSTM layers.
        h1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        h2 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c2 = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Initialize window mechanism variables.
        prev_kappa = torch.zeros(batch_size, self.window_mixtures, device=device)
        # Window vector: weighted sum over the one-hot encoded text.
        window_vec = torch.zeros(batch_size, self.char_vocab_size, device=device)
        
        outputs = []  # Collect MDN outputs over time

        # Process each time step.
        for t in range(seq_len):
            # Get the current pen stroke: shape (batch, input_dim)
            x_t = input_seq[:, t, :]
            # Concatenate pen stroke with previous window vector.
            lstm1_input = torch.cat([x_t, window_vec], dim=1)
            # Update LSTM1.
            h1, c1 = self.lstm1(lstm1_input, (h1, c1))
            
            # -------------------------
            # Compute window parameters from LSTM1's output.
            # -------------------------
            window_params = self.fc_window(h1)  # shape: (batch, 3*window_mixtures)
            window_params = window_params.view(batch_size, self.window_mixtures, 3)
            delta_kappa = torch.exp(window_params[:, :, 0])
            alpha = torch.exp(window_params[:, :, 1])
            beta = torch.exp(window_params[:, :, 2])
            # Update kappa (monotonically increasing).
            kappa = prev_kappa + delta_kappa
            prev_kappa = kappa

            # Compute the attention (phi) over text positions.
            u = torch.arange(0, max_text_len, device=device).float().view(1, 1, -1)  # (1,1,max_text_len)
            # Compute each component's contribution.
            phi = alpha.unsqueeze(2) * torch.exp(-beta.unsqueeze(2) * (kappa.unsqueeze(2) - u) ** 2)
            # Sum over window mixtures: (batch, max_text_len)
            phi = phi.sum(dim=1)
            # Compute new window vector: weighted sum of one-hot text vectors.
            window_vec = torch.bmm(phi.unsqueeze(1), text_encoded).squeeze(1)

            # -------------------------
            # LSTM2: Process the concatenation of LSTM1's output and the window vector.
            # -------------------------
            lstm2_input = torch.cat([h1, window_vec, x_t], dim=1)
            h2, c2 = self.lstm2(lstm2_input, (h2, c2))
            
            # -------------------------
            # MDN output: Predict mixture parameters from LSTM2's output.
            # -------------------------
            mdn_params = self.fc_mdn(h2)  # shape: (batch, 6*num_mixtures+1)
            outputs.append(mdn_params)

        mdn_params_seq = torch.stack(outputs, dim=1)  # (batch, seq_len, 6*num_mixtures+1)
        return mdn_params_seq

    def generate_step(self, x_t, hidden1, hidden2, prev_kappa, window_vec, text_encoded):
        """
        Generation step that processes one time step and returns attention weights.
        
        Args:
          x_t: (batch, input_dim)
          hidden1: Tuple (h1, c1) for LSTM1.
          hidden2: Tuple (h2, c2) for LSTM2.
          prev_kappa: (batch, window_mixtures)
          window_vec: (batch, char_vocab_size)
          text_encoded: (batch, max_text_len, char_vocab_size)
          
        Returns:
          mdn_params: MDN output, shape (batch, 6*num_mixtures+1)
          hidden1: Updated LSTM1 state.
          hidden2: Updated LSTM2 state.
          kappa: Updated kappa.
          window_vec: Updated window vector.
          phi: Attention weights over text positions, shape (batch, max_text_len)
        """
        batch_size = x_t.size(0)
        device = x_t.device
        max_text_len = text_encoded.size(1)

        # --- LSTM1 Update ---
        lstm1_input = torch.cat([x_t, window_vec], dim=1)
        h1, c1 = self.lstm1(lstm1_input, hidden1)

        # --- Window Mechanism ---
        window_params = self.fc_window(h1).view(batch_size, self.window_mixtures, 3)
        delta_kappa = torch.exp(window_params[:, :, 0])
        alpha = torch.exp(window_params[:, :, 1])
        beta = torch.exp(window_params[:, :, 2])
        kappa = prev_kappa + delta_kappa  # Monotonic update.


        # Compute attention over text positions, including an extra "end-of-text" token.
        # u now goes from 0 to max_text_len (i.e. U+1 positions)
        u = torch.arange(0, max_text_len + 1, device=device).float().view(1, 1, -1)
        phi_components = alpha.unsqueeze(2) * torch.exp(-beta.unsqueeze(2) * (kappa.unsqueeze(2) - u) ** 2)
        phi = phi_components.sum(dim=1)  # shape: (batch, max_text_len+1)
        
        # Compute new window vector using only the first max_text_len positions.
        window_vec = torch.bmm(phi[:, :-1].unsqueeze(1), text_encoded).squeeze(1)

        # --- LSTM2 Update ---
        lstm2_input = torch.cat([h1, window_vec, x_t], dim=1)
        h2, c2 = self.lstm2(lstm2_input, hidden2)

        # --- MDN Output ---
        mdn_params = self.fc_mdn(h2)

        return mdn_params, (h1, c1), (h2, c2), kappa, window_vec, phi

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
    # prob = torch.clamp(prob, min=1e-7, max=1-1e-7)

    # Negative log-likelihood for the coordinates.
    loss_mdn = -torch.log(prob)  # (B, T)
    # -------------------------
    # Pen state loss:
    # Use binary cross-entropy between the predicted pen probability and target.
    # -------------------------
    pen_prob = torch.clamp(pen_prob, min=1e-7, max=1-1e-7)
    loss_pen = F.binary_cross_entropy(pen_prob, pen_target, reduction='none')  # (B, T)

    # Total loss at each time step is the sum of the MDN loss and the pen loss.
    loss = loss_mdn + loss_pen

    # Average loss over all time steps and the batch.
    return torch.mean(loss)