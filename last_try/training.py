import model as model_def
import torch
import data_prep as data
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split

# =========================
# 3. Example Usage
# =========================

# Hyperparameters
input_dim = 3          # (x, y, pen state)
hidden_dim = 400       # hidden state size
num_mixtures = 20      # number of Gaussian mixtures in the MDN output
window_mixtures = 10   # number of mixtures for the window (attention) mechanism
epochs = 10000
char_vocab_size = len(model_def.vocab)

MODEL_PATH = "masked_models"

# Instantiate the model
model = model_def.HandwritingRNN(input_dim, hidden_dim, num_mixtures, char_vocab_size, window_mixtures)
# checkpoint = torch.load("last_models/epoch1271_train0.1182_val0.1379.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
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
# dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=data.handwriting_collate_fn)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Split dataset into 90% training and 20% validation.
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print("Train size: ",train_size)
print("Validation size: ",val_size)

# Create DataLoaders for training and validation.
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=data.handwriting_collate_fn)
val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=data.handwriting_collate_fn)

# Add learning rate scheduler 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

starter_epoch = 0

def load_dicts(path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    global starter_epoch
    starter_epoch = checkpoint['epoch']

# load_dicts("kappa_models/epoch154_train-2.9860_val-2.9017.pth")

best_val_loss = float('inf')


lambda_kappa = 0.1

for epoch in range(1,epochs):
    # ----- Training Phase -----
    model.train()
    total_train_loss = 0
    
    for i, (input_seq, target_seq, text) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Create a mask that identifies only complete zero vectors [0,0,0]
        # This checks if all values in each position are exactly zero
        padding_mask = ~torch.all(target_seq == 0, dim=2)  # Shape: [batch_size, seq_length]
        # The ~ operator inverts the mask, so True means "not padding" (i.e., keep this data point)
        
        # Forward pass: compute MDN parameters for the input sequence
        mdn_params_seq, kappas = model(input_seq, text)
        
        # Compute the unmasked MDN loss
        batch_losses = model_def.mdn_loss(mdn_params_seq, target_seq, num_mixtures)
        
        # Apply the mask and compute the proper average loss
        if batch_losses.dim() > padding_mask.dim():
            # If batch_losses has an extra dimension, sum over it first
            batch_losses = batch_losses.sum(dim=-1)
        
        # Apply the mask (set padded values to 0)
        masked_losses = batch_losses * padding_mask.float()
        
        # Compute the masked average (sum of masked losses divided by count of non-padded elements)
        num_non_padded = padding_mask.float().sum() + 1e-8  # Add small epsilon to avoid division by zero
        loss_mdn = masked_losses.sum() / num_non_padded
        
        # Compute kappa penalty with proper masking
        delta_kappa = kappas[:, 1:, :] - kappas[:, :-1, :]
        
        # For kappa penalty, only consider positions where both current and next step are non-padded
        kappa_mask = padding_mask[:, :-1] & padding_mask[:, 1:]
        masked_delta_kappa = delta_kappa * kappa_mask.unsqueeze(-1).float()
        
        # Compute the masked kappa penalty
        num_kappa_elements = kappa_mask.float().sum() + 1e-8
        kappa_penalty = lambda_kappa * (masked_delta_kappa.sum() / num_kappa_elements)
        
        # Total loss
        loss = loss_mdn + kappa_penalty
        
        # Backward pass and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_train_loss += loss.item()

        f = open("logi.txt", "a")
        print(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Kappa penalty: {kappa_penalty}")
        f.write(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Kappa penalty: {kappa_penalty}\n")
        f.close()
    
    avg_train_loss = total_train_loss / len(train_loader)

    f = open("logi.txt", "a")
    print(f"Epoch [{starter_epoch + epoch}/{epochs}], Training Loss: {avg_train_loss:.4f}")
    f.write(f"Epoch [{starter_epoch + epoch}/{epochs}], Training Loss: {avg_train_loss:.4f}\n")
    f.close()

    # ----- Validation Phase -----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (input_seq, target_seq, text) in enumerate(val_loader):
            # Create padding mask to identify only complete zero vectors
            padding_mask = ~torch.all(target_seq == 0, dim=2)
            
            # Forward pass
            mdn_params_seq, kappas_unused = model(input_seq, text)
            
            # Compute unmasked loss with reduction='none'
            batch_losses = model_def.mdn_loss(mdn_params_seq, target_seq, num_mixtures)
            
            # Apply mask and compute proper average
            if batch_losses.dim() > padding_mask.dim():
                batch_losses = batch_losses.sum(dim=-1)
            
            masked_losses = batch_losses * padding_mask.float()
            num_non_padded = padding_mask.float().sum() + 1e-8
            loss = masked_losses.sum() / num_non_padded
            
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)

    f = open("logi.txt", "a")
    print(f"Epoch [{starter_epoch + epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
    f.write(f"Epoch [{starter_epoch + epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}\n")
    f.close()
    
    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    # Save model checkpoint including both training and validation loss.
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Also save scheduler state
            'epoch': starter_epoch + epoch,
            'val_loss': avg_val_loss,
            }, MODEL_PATH + f"/epoch{starter_epoch + epoch}_train{avg_train_loss:.4f}_val{avg_val_loss:.4f}.pth")
    
    # Save best model separately
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': starter_epoch + epoch,
                'val_loss': avg_val_loss,
                }, MODEL_PATH + f"/best_model.pth")