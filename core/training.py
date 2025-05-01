import model as model_def
import torch
import data_prep as data
import os
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import argparse

# =========================
# 2. Args
# =========================

parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('lr', type=float,
                    help='Learning rate parameter')

parser.add_argument('optim', type=str,
                    help='adam or rms')

parser.add_argument('savefile', type=str,
                    help='checkpoint save file')

parser.add_argument('--opt_checkpoint', type=str,
                    help='checkpoint to load')

args = parser.parse_args()


# =========================
# 3. Example Usage
# =========================

# Hyperparameters

MODEL_PATH = args.savefile
LEARNING_RATE = args.lr
BATCH_SIZE = 64

input_dim = 3          # (x, y, pen state)
hidden_dim = 650       # hidden state size
num_mixtures = 10      # number of Gaussian mixtures in the MDN output
window_mixtures = 4    # number of mixtures for the window (attention) mechanism
epochs = 10000
char_vocab_size = len(model_def.vocab)

# Instantiate the model
model = model_def.HandwritingRNN(input_dim, hidden_dim, num_mixtures, char_vocab_size, window_mixtures)

# Assign the character dictionary to the model for use in text encoding.
model.char_to_idx = model_def.char_to_idx

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print('Using adam optimizer with ',LEARNING_RATE)
    f = open("logi.txt", "a")
    f.write(f"Using adam optimizer with {LEARNING_RATE}\n")
    f.close()

if args.optim == 'rms':
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, centered=True)
    print('Using rmsprop optimizer with ',LEARNING_RATE)
    f = open("logi.txt", "a")
    f.write(f"Using rms optimizer with {LEARNING_RATE}\n")
    f.close()

starter_epoch = 0

def load_dicts(path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    global starter_epoch
    starter_epoch = checkpoint['epoch']

    print('Loaded model from ',path)
    print('Epoch: ',starter_epoch)

    f = open("logi.txt", "a")
    f.write(f"Loaded model from {path}\n")
    f.write(f"Epoch: {starter_epoch}\n")
    f.close()

if args.opt_checkpoint is not None:
    full_path = args.opt_checkpoint
    print("Model path: ", full_path)
    f = open("logi.txt", "a")
    f.write(f"Model path:  {full_path}\n")
    f.close()
    load_dicts(full_path)

# Obsługa wielu folderów
folder_paths = ["../data/output", "../data/mwoutput", "../data/poloutput", "../data/hibru"]  # Lista ścieżek do folderów

# Funkcja do wczytywania danych z wielu folderów
def load_from_folders(folder_paths):
    all_svg_files = []
    all_text_files = []
    
    for folder_path in folder_paths:
        # Ścieżka do pliku z tekstami dla bieżącego folderu
        files_content = f"{folder_path}/files.txt"
        
        # Lista nazw plików z rozszerzeniem .svg z bieżącego folderu
        svg_files = [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith('.svg')]
        
        # Sortowanie plików SVG numerycznie
        svg_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        all_svg_files.extend(svg_files)
        all_text_files.append(files_content)
    
    return all_svg_files, all_text_files

# Wczytanie danych z wielu folderów
svg_files, text_files = load_from_folders(folder_paths)

dataset = data.HandwritingDataset(svg_files, text_files)

# Split dataset into 90% training and 20% validation.
dataset_size = len(dataset)
train_size = int(0.95 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print("Train size: ",train_size)
print("Validation size: ",val_size)

# Add learning rate scheduler 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7)

# Create DataLoaders for training and validation.
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data.handwriting_collate_fn)
val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data.handwriting_collate_fn)

best_val_loss = float('inf')

special_chars = set('ąężźćńłóśABCĆDEFGHIJKLŁMNOÓPQRSŚTUVWXYZŹŻ0123456789\'-!\"#$%&()*,./:;?@[]+<=>qQvVxX')

raw_counts = {
    'a': 10559,
    'ą': 491,
    'b': 1900,
    'c': 4412,
    'ć': 386,
    'd': 3824,
    'e': 10912,
    'b': 1900,
    'c': 4412,
    'ć': 386,
    'd': 3824,
    'e': 10912,
    'ę': 975,
    'f': 266,
    'g': 1413,
    'h': 989,
    'i': 9747,
    'j': 2774,
    'k': 3302,
    'l': 3776,
    'ł': 1068,
    'm': 4288,
    'n': 5710,
    'ń': 153,
    'o': 9177,
    'ó': 428,
    'p': 3171,
    'q': 28,
    'r': 4741,
    's': 5944,
    'ś': 476,
    't': 4553,
    'u': 2790,
    'v': 78,
    'w': 3999,
    'x': 67,
    'y': 4317,
    'z': 7169,
    'ź': 177,
    'ż': 466,
    'A': 136,
    'B': 139,
    'C': 420,
    'Ć': 19,
    'D': 161,
    'E': 109,
    'F': 117,
    'G': 119,
    'H': 119,
    'I': 137,
    'J': 297,
    'K': 158,
    'L': 122,
    'Ł': 103,
    'M': 456,
    'N': 381,
    'O': 222,
    'Ó': 18,
    'P': 405,
    'Q': 14,
    'R': 115,
    'S': 232,
    'Ś': 102,
    'T': 601,
    'U': 124,
    'V': 20,
    'W': 300,
    'X': 20,
    'Y': 78,
    'Z': 229,
    'Ź': 29,
    'Ż': 102,
    ' ': 16708,
    '0': 248,
    '1': 126,
    '2': 93,
    '3': 77,
    '4': 58,
    '5': 77,
    '6': 56,
    '7': 58,
    '8': 60,
    '9': 71,
    '!': 77,
    '"': 83,
    '#': 26,
    '%': 26,
    '&': 26,
    "'": 52,
    '(': 25,
    ')': 25,
    '*': 26,
    '+': 25,
    ',': 293,
    '-': 36,
    '.': 200,
    '/': 28,
    ':': 37,
    ';': 26,
    '<': 25,
    '=': 25,
    '>': 25,
    '?': 169,
    '@': 25,
    '[': 26,
    ']': 26,
    '$': 26
}

avg_count = 1212

# Option A: simple inverse frequency
char_weights = {c: 1.0 / cnt for c, cnt in raw_counts.items()}


max_count = max(raw_counts.values())
char_weights = {c: avg_count / cnt for c, cnt in raw_counts.items()}

# L1 normalization
# total = sum(char_weights.values())
# char_weights = {c: w / total for c, w in char_weights.items()}
# print(char_weights)
# input("")

for epoch in range(starter_epoch + 1, epochs):
    # ----- Training Phase -----
    model.train()
    total_train_loss = 0
    start_time = time.time()
    for i, (input_seq, target_seq, text) in enumerate(train_loader):
        optimizer.zero_grad()

        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # Create a mask that identifies only complete zero vectors [0,0,0]
        # This checks if all values in each position are exactly zero
        padding_mask = ~torch.all(target_seq == 0, dim=2)  # Shape: [batch_size, seq_length]
        # The ~ operator inverts the mask, so True means "not padding" (i.e., keep this data point)
        
        # Forward pass: compute MDN parameters for the input sequence
        mdn_params_seq, window_params, phi = model(input_seq, text)
        
        # Compute the unmasked MDN loss
        batch_losses = model_def.mdn_loss(mdn_params_seq, target_seq, num_mixtures)
        
        # Apply the mask and compute the proper average loss
        if batch_losses.dim() > padding_mask.dim():
            # If batch_losses has an extra dimension, sum over it first
            batch_losses = batch_losses.sum(dim=-1)
        
        # Apply the mask (set padded values to 0)
        masked_losses = batch_losses * padding_mask.float()

        batch_size, seq_len = masked_losses.shape


        # focused_char_idx = phi.argmax(dim=2)  # Shape: [batch_size, seq_len]

        # weights_per_timestep = torch.ones_like(focused_char_idx, dtype=torch.float32, device=device)

        # # 2. Build a tensor of char weights
        # max_text_len = max(len(txt) for txt in text)
        # char_weight_tensor = torch.ones((len(text), max_text_len), device=device)

        # for k, text_seq in enumerate(text):
        #     for j, ch in enumerate(text_seq):
        #         char_weight_tensor[k, j] = char_weights.get(ch, 1.0)

        # # 3. Lookup weights per timestep
        # weights_per_timestep = torch.gather(char_weight_tensor, 1, focused_char_idx)

        # scaled_losses = masked_losses * weights_per_timestep

        # Build a tensor of W for each sample
        W = torch.ones(batch_size, device=device, dtype=torch.float32)

        for j, text in enumerate(text):
            # sum weights for each character, ignoring padding; then average
            weights = [ char_weights.get(ch, 1.0) for ch in text ]
            if len(weights) > 0:
                W[j] = sum(weights) / len(weights)
            else:
                W[j] = 1.0
            
            if(len(text) > 30):
                W[j] += 1
            
            
            # print(text,W)
            # input("")


        # Expand to [batch_size, seq_len] and apply
        W_expanded = W.unsqueeze(1)            # [batch_size, 1]
        scaled_losses = masked_losses * W_expanded
        
        # # Count special characters in each text and use it as a multiplier
        # scale_factors = torch.ones(len(text), device=device)
        # for idx, text in enumerate(text):
        #     count = sum(char in special_chars for char in text)
        #     if count > 0:

        # Final average loss over non-padded tokens
        num_non_padded = padding_mask.float().sum() + 1e-8
        loss_mdn = scaled_losses.sum() / num_non_padded

        # Compute the masked average (sum of masked losses divided by count of non-padded elements)
        # num_non_padded = padding_mask.float().sum() + 1e-8  # Add small epsilon to avoid division by zero
        # loss_mdn = masked_losses.sum() / num_non_padded

        # Extract parameters after exponential
        
        # [Batch_size, seq_len, window_mixtures, (a,b,k)]  
        log_kappa = window_params[:, :, :, 0]
        log_alpha = window_params[:, :, :, 1]
        log_beta = window_params[:, :, :, 2]

        # # Define thresholds
        alpha_min, alpha_max = 1, 10
        beta_min, beta_max = 0.4, 10 
        kappa_min, kappa_max = 0.03, 0.05 

        # # Compute penalties
        alpha_penalty = (torch.relu(alpha_min - log_alpha) + torch.relu(log_alpha - alpha_max)) * padding_mask.float().unsqueeze(-1) 
        beta_penalty = (torch.relu(beta_min - log_beta) + torch.relu(log_beta - beta_max)) * padding_mask.float().unsqueeze(-1)

        # # Sum the penalties to get a loss term
        penalizing_loss = (alpha_penalty.sum() + beta_penalty.sum()) / num_non_padded

        # # Compute kappa penalty with proper masking
        kappa_penalty = (torch.relu(kappa_min - log_kappa) + torch.relu(log_kappa - kappa_max)) * padding_mask.float().unsqueeze(-1)

        penalizing_loss += (kappa_penalty.sum() / num_non_padded)

        # Total loss
        loss = loss_mdn + penalizing_loss

        # loss = loss_mdn
        
        # Backward pass and optimization
        loss.backward()

        

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()

        
        
        total_train_loss += loss.item()

        f = open("logi.txt", "a")
        print(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        f.write(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}\n")
        f.close()
        # print("Alpha mean:", log_alpha.mean().item(), "Alpha max:", log_alpha.max().item(), "Alpha min:", log_alpha.min().item())
        # print("Beta mean:", log_beta.mean().item(), "Beta max:", log_beta.max().item(), "Beta min:", log_beta.min().item())
        # print("Kappa mean:", log_kappa.mean().item(), "Kappa max:", log_kappa.max().item(), "Kappa min:", log_kappa.min().item())
        # print("Phi mean:", phi.mean().item(), "Phi max:", phi.max().item(), "Phi min:", phi.min().item())

    
    avg_train_loss = total_train_loss / len(train_loader)

    f = open("logi.txt", "a")
    print(f"Epoch [{epoch}/{epochs}], Training Loss: {avg_train_loss:.4f}")
    f.write(f"Epoch [{epoch}/{epochs}], Training Loss: {avg_train_loss:.4f}\n")
    f.close()

    # ----- Validation Phase -----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (input_seq, target_seq, text) in enumerate(val_loader):

            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # Create padding mask to identify only complete zero vectors
            padding_mask = ~torch.all(target_seq == 0, dim=2)
            
            # Forward pass
            mdn_params_seq, window_unused, phi_unused = model(input_seq, text)
            
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
    end_time = time.time()
    elapsed_time = end_time - start_time

    f = open("logi.txt", "a")
    print(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
    print(f"Time: {elapsed_time:.2f} seconds")
    f.write(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}\n")
    f.close()
    
    # Step the scheduler based on validation loss
    # scheduler.step(avg_val_loss)
    
    # Save model checkpoint including both training and validation loss.
    if(epoch % 1 == 0):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),  # Also save scheduler state
            'epoch': epoch,
            'val_loss': avg_val_loss,
            }, MODEL_PATH + f"/epoch{epoch}_train{avg_train_loss:.4f}_val{avg_val_loss:.4f}.pth")
    
    # Save best model separately
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    }, MODEL_PATH + f"/best_model.pth")
