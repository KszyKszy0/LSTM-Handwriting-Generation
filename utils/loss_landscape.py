import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import last_try.model as model_def
import last_try.data_prep as data
import os 

# Assume you have a trained model
# Hyperparameters
input_dim = 3          # (x, y, pen state)
hidden_dim = 400       # hidden state size
num_mixtures = 20      # number of Gaussian mixtures in the MDN output
window_mixtures = 10   # number of mixtures for the window (attention) mechanism
epochs = 10000
char_vocab_size = len(model_def.vocab)

# Instantiate the model
model = model_def.HandwritingRNN(input_dim, hidden_dim, num_mixtures, char_vocab_size, window_mixtures)
# checkpoint = torch.load("last_models/epoch1271_train0.1182_val0.1379.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# Assign the character dictionary to the model for use in text encoding.
model.char_to_idx = model_def.char_to_idx

model.load_state_dict(torch.load("rms_prop3\epoch496_train0.4465_val1.9521.pth")['model_state_dict'])  # Load trained weights

# Obsługa wielu folderów
folder_paths = ["mwoutput", "output"]  # Lista ścieżek do folderów

# Funkcja do wczytywania danych z wielu folderów
def load_from_folders(folder_paths):
    all_svg_files = []
    all_text_files = []
    
    for folder_path in folder_paths:
        # Ścieżka do pliku z tekstami dla bieżącego folderu
        files_content = f"{folder_path}/files.txt"
        
        # Lista nazw plików z rozszerzeniem .svg z bieżącego folderu
        svg_files = [f"{folder_path}/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]
        
        all_svg_files.extend(svg_files)
        all_text_files.append(files_content)
    
    return all_svg_files, all_text_files

# Wczytanie danych z wielu folderów
svg_files, text_files = load_from_folders(folder_paths)

dataset = data.HandwritingDataset(svg_files, text_files)

# Get model parameters as a single tensor
params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()

# Generate two random direction vectors
d1 = torch.randn_like(params)
d2 = torch.randn_like(params)

# Normalize directions
d1 /= torch.norm(d1)
d2 /= torch.norm(d2)

# Define interpolation range
alphas = np.linspace(-1, 1, 4)
betas = np.linspace(-1, 1, 4)

# Store loss values
loss_values = np.zeros((len(alphas), len(betas)))

# Use a small subset of data
small_dataset = torch.utils.data.Subset(dataset, indices=range(int(len(dataset) * 0.05)))
small_dataloader = torch.utils.data.DataLoader(small_dataset, batch_size=32, shuffle=False, collate_fn=data.handwriting_collate_fn)

# Compute loss landscape
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        # Modify model parameters
        new_params = params + alpha * d1 + beta * d2
        torch.nn.utils.vector_to_parameters(new_params, model.parameters())

        # Compute loss on small dataset
        loss = 0.0
        model.eval()
        with torch.no_grad():
            for z, (input_seq, target_seq, text) in enumerate(small_dataloader):
                
                padding_mask = ~torch.all(target_seq == 0, dim=2)  # Shape: [batch_size, seq_length]

                # Forward pass: compute MDN parameters for the input sequence
                mdn_params_seq, kappas = model(input_seq, text)
                
                # Compute the unmasked MDN loss
                batch_losses = model_def.mdn_loss(mdn_params_seq, target_seq, num_mixtures)

                if batch_losses.dim() > padding_mask.dim():
                    # If batch_losses has an extra dimension, sum over it first
                    batch_losses = batch_losses.sum(dim=-1)
                
                # Apply the mask (set padded values to 0)
                masked_losses = batch_losses * padding_mask.float()
                
                # Compute the masked average (sum of masked losses divided by count of non-padded elements)
                num_non_padded = padding_mask.float().sum() + 1e-8  # Add small epsilon to avoid division by zero
                loss_mdn = masked_losses.sum() / num_non_padded
                
                loss += loss_mdn.item()

        loss_values[i, j] = loss / len(small_dataloader)

        print(i*len(alphas) + j," / ",len(alphas)*len(betas))

# Plot loss landscape
X, Y = np.meshgrid(alphas, betas)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, loss_values, cmap='viridis')
ax.set_xlabel("Direction 1")
ax.set_ylabel("Direction 2")
ax.set_zlabel("Loss")
plt.show()