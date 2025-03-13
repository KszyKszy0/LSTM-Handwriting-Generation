import torch
import torch.nn.functional as F
import numpy as np
import svgwrite
import model as model_def

# Hyperparameters
input_dim = 3          # (x, y, pen state)
hidden_dim = 500       # hidden state size
num_layers = 3         # number of LSTM layers
num_mixtures = 4      # number of Gaussian mixtures in the MDN output
window_mixtures = 2   # number of mixtures for the window (attention) mechanism
epochs = 100
char_vocab_size = len(model_def.vocab)

# =========================
# 2. Define a function to sample from the MDN output.
# =========================
def sample_from_mdn(mdn_params, num_mixtures, temperature=1.0):
    """
    Sample a stroke (x, y, pen state) from the MDN parameters.
    
    Args:
      mdn_params: Tensor of shape (1, 6*num_mixtures+1) (batch size is 1).
      num_mixtures: Number of mixtures used.
      temperature: Sampling temperature to control randomness.
      
    Returns:
      x_sample, y_sample, pen_sample: Sampled stroke values.
    """
    # Split parameters.
    pi = mdn_params[:, :num_mixtures]                          # (1, M)
    mu1 = mdn_params[:, num_mixtures:2*num_mixtures]
    mu2 = mdn_params[:, 2*num_mixtures:3*num_mixtures]
    sigma1 = torch.exp(mdn_params[:, 3*num_mixtures:4*num_mixtures])
    sigma2 = torch.exp(mdn_params[:, 4*num_mixtures:5*num_mixtures])
    rho = torch.tanh(mdn_params[:, 5*num_mixtures:6*num_mixtures])
    pen_logit = mdn_params[:, -1]
    
    # Apply activations.
    pi = F.softmax(pi / temperature, dim=-1).cpu().numpy().flatten()
    pen_prob = torch.sigmoid(pen_logit).item()
    
    # Choose a mixture component.
    mixture_idx = np.random.choice(np.arange(num_mixtures), p=pi)
    mu1_val = mu1[0, mixture_idx].item()
    mu2_val = mu2[0, mixture_idx].item()
    sigma1_val = sigma1[0, mixture_idx].item() * temperature
    sigma2_val = sigma2[0, mixture_idx].item() * temperature
    rho_val = rho[0, mixture_idx].item()
    
    # Sample from a bivariate Gaussian with correlation.
    u = np.random.randn()
    v = np.random.randn()
    x_sample = mu1_val + sigma1_val * u
    y_sample = mu2_val + sigma2_val * (rho_val * u + np.sqrt(1 - rho_val**2) * v)
    
    # Sample pen state: if random number < pen_prob then pen is lifted (1), else pen down (0).
    pen_sample = 1.0 if np.random.rand() < pen_prob else 0.0
    
    return x_sample, y_sample, pen_sample



# =========================
# 3. Handwriting Generation Function
# =========================
def generate_handwriting(model, text, seq_len=300, temperature=1.0):
    """
    Generate a handwriting sequence for a given text string.
    
    Args:
      model: Trained HandwritingRNN.
      text: Text string to condition on.
      seq_len: Number of time steps to generate.
      temperature: Sampling temperature.
      
    Returns:
      strokes: List of (x, y, pen_state) tuples.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        text_encoded, _ = model.encode_text_batch([text])
        text_encoded = text_encoded.to(device)
        batch_size = 1

        # Initialize hidden states for both LSTM layers.
        h1 = torch.zeros(batch_size, model.hidden_dim, device=device)
        c1 = torch.zeros(batch_size, model.hidden_dim, device=device)
        hidden1 = (h1, c1)
        h2 = torch.zeros(batch_size, model.hidden_dim, device=device)
        c2 = torch.zeros(batch_size, model.hidden_dim, device=device)
        hidden2 = (h2, c2)

        prev_kappa = torch.zeros(batch_size, model.window_mixtures, device=device)
        window_vec = torch.zeros(batch_size, model.char_vocab_size, device=device)
        current_input = torch.zeros(batch_size, model.input_dim, device=device)
        strokes = []
        
        for t in range(seq_len):
            mdn_params, hidden1, hidden2, prev_kappa, window_vec = model.generate_step(
                current_input, hidden1, hidden2, prev_kappa, window_vec, text_encoded)
            x_sample, y_sample, pen_sample = sample_from_mdn(mdn_params, model.num_mixtures, temperature)
            current_input = torch.tensor([[x_sample, y_sample, pen_sample]], device=device, dtype=torch.float32)
            strokes.append((x_sample, y_sample, pen_sample))
            
        return strokes
    

# =========================
# 4. Function to Save Generated Strokes as an SVG File
# =========================
def save_strokes_to_svg(strokes, filename, scale=1.0, stroke_width=2):
    """
    Save a list of strokes to an SVG file.
    
    Args:
      strokes: List of (x, y, pen_state) tuples. Here, (x, y) are stroke deltas.
      filename: Path to the SVG file to save.
      scale: Scaling factor to make the drawing larger.
      stroke_width: Width of the stroke in the SVG.
    """
    # Convert stroke deltas to absolute coordinates.
    x, y = 50, 50
    drawing = svgwrite.Drawing(filename, profile='tiny')
    current_path = ""
    
    for dx, dy, pen in strokes:
        # Update absolute positions.
        x += dx
        y += dy
        # When pen is down (pen < 0.5), continue drawing.
        if pen < 0.5:
            if current_path == "":
                current_path += f"M {x*scale} {y*scale} "
            else:
                current_path += f"L {x*scale} {y*scale} "
        else:
            # When pen is up, finish the current path if it exists.
            if current_path != "":
                drawing.add(drawing.path(d=current_path, fill="none", stroke="black", stroke_width=stroke_width))
                current_path = ""
    # Add any remaining path.
    if current_path != "":
        drawing.add(drawing.path(d=current_path, fill="none", stroke="black", stroke_width=stroke_width))
    drawing.save()
    print(f"SVG saved to {filename}")



# =========================
# 5. Load Model and Generate Handwriting for a Given String
# =========================
def load_model_and_generate(model_path, text, seq_len=300, output_svg="output.svg", temperature=1.0):
    """
    Load a saved model and generate handwriting for the provided text.
    The generated handwriting is saved as an SVG file.
    
    Args:
      model_path: Path to the saved model file (e.g. 'model.pth').
      text: The text string to generate handwriting for.
      seq_len: Number of time steps to generate.
      output_svg: Output filename for the SVG.
      temperature: Sampling temperature.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model.
    # model = torch.load(model_path, map_location=device)
    model = model_def.HandwritingRNN(input_dim, hidden_dim, num_mixtures, char_vocab_size, window_mixtures)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # It is assumed that the model's character dictionary has been saved externally.
    # If not, set it here.
    if model.char_to_idx is None:
        # Example vocabulary: letters and space.
        vocab = model_def.vocab
        model.char_to_idx = {c: i for i, c in enumerate(vocab)}
    
    # Generate handwriting strokes.
    strokes = generate_handwriting(model, text, seq_len, temperature)
    # Save the strokes as an SVG.
    save_strokes_to_svg(strokes, output_svg)


# =========================
# Example usage:
# Uncomment and modify the following lines to load your model and generate handwriting.
model_path = "last_models/epoch65_train2.5780_val2.7817.pth"       # path to your saved model file
text_to_generate = "teraz test"
load_model_and_generate(model_path, text_to_generate, seq_len=200, output_svg="handwriting.svg", temperature=0.8)