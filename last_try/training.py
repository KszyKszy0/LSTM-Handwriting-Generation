import model as model_def
import torch
import data_prep as data
import os
import torch.optim as optim
import torch.nn as nn

# =========================
# 3. Example Usage
# =========================

# Hyperparameters
input_dim = 3          # (x, y, pen state)
hidden_dim = 80       # hidden state size
num_layers = 3         # number of LSTM layers
num_mixtures = 20      # number of Gaussian mixtures in the MDN output
window_mixtures = 10   # number of mixtures for the window (attention) mechanism
epochs = 10000
char_vocab_size = len(model_def.vocab)

MODEL_PATH = "last_models"

# Instantiate the model
model = model_def.HandwritingRNN(input_dim, hidden_dim, num_layers, num_mixtures, char_vocab_size, window_mixtures)
# Assign the character dictionary to the model for use in text encoding.
model.char_to_idx = model_def.char_to_idx

# Dummy input:
# Let's assume a batch size of 2 and sequence length of 50 time steps.
# batch_size = 2
# seq_len = 50
# dummy_input = torch.randn(batch_size, seq_len, input_dim)  # random pen stroke data

# # Dummy target (for training, same shape as input):
# dummy_target = torch.randn(batch_size, seq_len, input_dim)

# # Dummy text: list of strings corresponding to each sample in the batch.
# dummy_text = ["hello", "world"]

# # Run the model forward pass.
# mdn_outputs = model(dummy_input, dummy_text)
# print("MDN output shape:", mdn_outputs.shape)
# Expected shape: (batch_size, seq_len, 6*num_mixtures + 1)


# Ścieżka do folderu
folder_path = "mwoutput"
files_content = f"{folder_path}/files.txt"
# Lista nazw plików z rozszerzeniem .svg
svg_files = [f"{folder_path}/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]

dataset = data.HandwritingDataset(svg_files, files_content)
dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=data.handwriting_collate_fn)

optimizer = optim.Adam(model.parameters(), lr=5e-6)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (input_seq, target_seq, text) in enumerate(dataloader):

        optimizer.zero_grad()

        # Forward pass: compute MDN parameters for the input sequence.
        mdn_params_seq = model(input_seq, text)  # shape: (B, T, 6*num_mixtures+1)

        # Compute the loss using our MDN loss function.
        loss = model_def.mdn_loss(mdn_params_seq, target_seq, num_mixtures)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        print(f"Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH + f"/epoch{epoch+1}_loss{avg_loss:.4f}.pth")