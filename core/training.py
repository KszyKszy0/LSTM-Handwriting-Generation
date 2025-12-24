import math
import requests
from sendimg import run_generate_and_upload
import model as model_def
import torch
import data_prep as data
import os
import re
import sys
from pathlib import Path
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import argparse
import argcomplete
from argcomplete.completers import DirectoriesCompleter, ChoicesCompleter
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter


# Dodaj katalog główny projektu do PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.get_losses import main as get_losses_main


def getDirs(__file__):
    cwdir = os.path.abspath(os.getcwd())
    cwdir += "/"
    filedir = os.path.abspath(os.path.dirname(__file__))
    filedir += "/"
    return cwdir, filedir

# =========================
# 2. Args
# =========================

parser = argparse.ArgumentParser(description='Training script for handwriting generation model.')

parser.add_argument('--lr', type=float,
                    help='Learning rate parameter')

parser.add_argument('--optim', type=str,
                    help='adam or rms', choices=['adam', 'rms'])

parser.add_argument('--savefile', type=str,
                    help='folder name for checkpoint save file').completer = DirectoriesCompleter()

parser.add_argument('--opt_checkpoint', type=str,
                    help='checkpoint file to load').completer = DirectoriesCompleter()

parser.add_argument("--batch_size", type=int)

parser.add_argument("--resume", action="store_true",)

argcomplete.autocomplete(parser)
args = parser.parse_args()

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


# =========================
# 3. Example Usage
# =========================

# Hyperparameters

cwdir, filedir = getDirs(__file__)

load_dotenv()

savefile = os.getenv("SAVEFILE")
if args.savefile is not None:
    savefile = args.savefile
MODEL_PATH = os.path.abspath(cwdir + savefile)

# === TensorBoard ===
LOG_DIR = "runs/" + savefile
writer = SummaryWriter(LOG_DIR)

LEARNING_RATE = float(os.getenv("LR"))
if args.lr is not None:
    LEARNING_RATE = args.lr
    
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

input_dim = 3          # (x, y, pen state)
hidden_dim = int(os.getenv("hidden_dim"))               # hidden state size
num_mixtures = int(os.getenv("num_mixtures"))           # number of Gaussian mixtures in the MDN output
window_mixtures = int(os.getenv("window_mixtures"))     # number of mixtures for the window (attention) mechanism
epochs = 10000
char_vocab_size = len(model_def.vocab)
logsFile = os.path.abspath(filedir + "../output/logi.txt")
website_url = f'https://{os.getenv("username")}:{os.getenv("password")}@ai-dashboard.d0d0.ovh'

# Instantiate the model
model = model_def.HandwritingRNN(input_dim, hidden_dim, num_mixtures, char_vocab_size, window_mixtures)

# Assign the character dictionary to the model for use in text encoding.
model.char_to_idx = model_def.char_to_idx

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

optimal = os.getenv("OPTIM")
if args.optim is not None:
    optimal = args.optim
    

    

if optimal == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print('Using adam optimizer with ',LEARNING_RATE)
    f = open(logsFile, "a")
    f.write(f"Using adam optimizer with {LEARNING_RATE}\n")
    f.close()

if optimal == 'rms':
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, centered=True)
    print('Using rmsprop optimizer with ',LEARNING_RATE)
    f = open(logsFile, "a")
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

    f = open(logsFile, "a")
    f.write(f"Loaded model from {path}\n")
    f.write(f"Epoch: {starter_epoch}\n")
    f.close()
    
if args.resume is not None:
    files = os.listdir(MODEL_PATH)
    
    regex = r"epoch(\d+)_.*"
    filtered_files = [(int(re.match(regex, file).group(1)), file) for file in files if re.match(regex, file)]

    if filtered_files:
        highest_epoch_file = max(filtered_files, key=lambda x: x[0])[1]
        print("File with the highest epoch:", highest_epoch_file)
        args.opt_checkpoint = highest_epoch_file
    else:
        print("No matching files found.")
    
    

if args.opt_checkpoint is not None:
    if args.resume is not None:
        full_path = os.path.abspath(MODEL_PATH + "/" + args.opt_checkpoint)
    else:
        full_path = os.path.abspath(cwdir + args.opt_checkpoint)
    print("Model path: ", full_path)
    f = open(logsFile, "a")
    f.write(f"Model path:  {full_path}\n")
    f.close()
    load_dicts(full_path)
    
if args.batch_size is not None:
    BATCH_SIZE = args.batch_size
    print("Batch size: ",BATCH_SIZE)
    f = open(logsFile, "a")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.close()

# Obsługa wielu folderów
folder_paths = ["../data/output", "../data/mwoutput", "../data/poloutput", "../data/hibru", "../data/augmented"]  # Lista ścieżek do folderów

# Funkcja do wczytywania danych z wielu folderów
def load_from_folders(folder_paths):
    all_svg_files = []
    all_text_files = []
    
    for folder_path in folder_paths:
        # Ścieżka do pliku z tekstami dla bieżącego folderu
        folder_path = os.path.abspath(filedir + folder_path)
        files_content = f"{folder_path}/files.txt"
        
        # Lista nazw plików z rozszerzeniem .svg z bieżącego folderu
        print("Loading folder: ",files_content)
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
    'a': 11762,
    'ą': 984,
    'b': 2153,
    'c': 5030,
    'ć': 688,
    'd': 4312,
    'e': 11981,
    'ę': 1716,
    'ą': 984,
    'b': 2153,
    'c': 5030,
    'ć': 688,
    'd': 4312,
    'e': 11981,
    'ę': 1716,
    'f': 300,
    'g': 1640,
    'h': 1122,
    'i': 11126,
    'j': 3094,
    'k': 3883,
    'l': 4021,
    'ł': 1897,
    'm': 4848,
    'n': 6327,
    'ń': 238,
    'o': 10236,
    'ó': 780,
    'p': 3655,
    'q': 150,
    'r': 5243,
    's': 6660,
    'ś': 903,
    't': 5023,
    'u': 3103,
    'v': 150,
    'w': 4555,
    'x': 151,
    'y': 4972,
    'z': 7992,
    'ź': 240,
    'ż': 955,
    'A': 177,
    'B': 198,
    'C': 510,
    'Ć': 150,
    'D': 202,
    'E': 159,
    'F': 154,
    'G': 160,
    'H': 155,
    'I': 167,
    'J': 335,
    'K': 200,
    'L': 162,
    'Ł': 151,
    'M': 563,
    'N': 504,
    'O': 260,
    'Ó': 150,
    'P': 502,
    'Q': 150,
    'R': 161,
    'S': 278,
    'Ś': 159,
    'T': 650,
    'U': 168,
    'V': 155,
    'W': 365,
    'X': 152,
    'Y': 150,
    'Z': 287,
    'Ź': 150,
    'Ż': 158,
    ' ': 19608,
    '0': 250,
    '1': 152,
    '2': 152,
    '3': 150,
    '4': 150,
    '5': 150,
    '6': 150,
    '7': 150,
    '8': 152,
    '9': 151,
    '!': 169,
    '"': 180,
    '#': 150,
    '%': 150,
    '&': 150,
    "'": 151,
    '(': 150,
    ')': 150,
    '*': 150,
    '+': 150,
    ',': 507,
    '-': 153,
    '.': 679,
    '/': 154,
    ':': 151,
    ';': 150,
    '<': 150,
    '=': 150,
    '>': 150,
    '?': 262,
    '@': 150,
    '[': 150,
    ']': 150,
    '$': 150
}

avg_count = 1445

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
        # W = torch.ones(batch_size, device=device, dtype=torch.float32)

        # for j, text in enumerate(text):
        #     # sum weights for each character, ignoring padding; then average
        #     weights = [ char_weights.get(ch, 1.0) for ch in text ]
        #     if len(weights) > 0:
        #         W[j] = sum(weights) / len(weights)
        #     else:
        #         W[j] = 1.0
            
        #     if(len(text) > 30):
        #         W[j] += 1
            
            
            # print(text,W)
            # input("")


        # Expand to [batch_size, seq_len] and apply
        # W_expanded = W.unsqueeze(1)            # [batch_size, 1]
        # scaled_losses = masked_losses * W_expanded
        
        # # Count special characters in each text and use it as a multiplier
        # scale_factors = torch.ones(len(text), device=device)
        # for idx, text in enumerate(text):
        #     count = sum(char in special_chars for char in text)
        #     if count > 0:

        # Final average loss over non-padded tokens
        # num_non_padded = padding_mask.float().sum() + 1e-8
        # loss_mdn = scaled_losses.sum() / num_non_padded

        # Compute the masked average (sum of masked losses divided by count of non-padded elements)
        num_non_padded = padding_mask.float().sum() + 1e-8  # Add small epsilon to avoid division by zero
        loss_mdn = masked_losses.sum() / num_non_padded

        # Extract parameters after exponential
        
        # [Batch_size, seq_len, window_mixtures, (a,b,k)] 
        if(epoch < 2): 
            log_kappa = window_params[:, :, :, 0]
            log_alpha = window_params[:, :, :, 1]
            log_beta = window_params[:, :, :, 2]

            # # Define thresholds
            alpha_min, alpha_max = 9, 11
            beta_min, beta_max = 2, 10 
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
            
        else:
            # If the epoch is less than 50, use only the MDN loss
            loss = loss_mdn
        
        # Backward pass and optimization
        loss.backward()

        

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        optimizer.step()

        
        
        total_train_loss += loss.item()

        f = open(logsFile, "a")
        print(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        f.write(f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}\n")
        f.close()
        # print("Alpha mean:", log_alpha.mean().item(), "Alpha max:", log_alpha.max().item(), "Alpha min:", log_alpha.min().item())
        # print("Beta mean:", log_beta.mean().item(), "Beta max:", log_beta.max().item(), "Beta min:", log_beta.min().item())
        # print("Kappa mean:", log_kappa.mean().item(), "Kappa max:", log_kappa.max().item(), "Kappa min:", log_kappa.min().item())
        # print("Phi mean:", phi.mean().item(), "Phi max:", phi.max().item(), "Phi min:", phi.min().item())

    
    avg_train_loss = total_train_loss / len(train_loader)

    f = open(logsFile, "a")
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

    f = open(logsFile, "a")
    print(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
    print(f"Time: {elapsed_time:.2f} seconds")
    f.write(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}\n")
    f.close()

    # Log error
    writer.add_scalars("Loss", {
        'Train': avg_train_loss,
        'Val': avg_val_loss
    }, epoch)
    
    # Step the scheduler based on validation loss
    # scheduler.step(avg_val_loss)
    
    # Save model checkpoint including both training and validation loss.
    if((epoch % 1 == 0)):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),  # Also save scheduler state
            'epoch': epoch,
            'val_loss': avg_val_loss,
            }, MODEL_PATH + f"/epoch{epoch}_train{avg_train_loss:.4f}_val{avg_val_loss:.4f}_batch{BATCH_SIZE}.pth")
    
    if(epoch % 10 == 0):
        get_losses_main(os.path.abspath(MODEL_PATH), printLog=False)
        
    #save to website
    if(epoch % 1 == 0):
        try:
            # check if loss in NaN
            if math.isnan(avg_train_loss) or math.isnan(avg_val_loss):
                print("NaN loss detected, skipping website update.")
                json = {
                    "model_name": os.getenv("NAME"),
                    "error": "NaN"
                }
                res = requests.post(website_url+"/error", json=json)
                if res.status_code != 200:
                    print(f"Błąd: {response.status_code}")
                print(response.text)
                continue
            
            train_loss = round(avg_train_loss, 4)
            val_loss = round(avg_val_loss, 4)
            temp_time = round(elapsed_time, 2)
            
            payload = {
                "model_name": os.getenv("NAME"),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": temp_time,
                "batch_size": BATCH_SIZE
            }
            response = requests.post(website_url+"/update", json=payload)

            if response.status_code != 200:
                print(f"Błąd: {response.status_code}")
                print(response.text)
        except:
            print("Nie ma połączenia z dashboardem")
      
    if(epoch % 15 == 0):
        # gen habdwriting and send to server
        print("Wysyłam zdjęcie do dashboardu")
        try:
            name = os.getenv("NAME")
            run_generate_and_upload(name, epoch)
        except:
            print("Błąd w wysyłaniu zdjęcia lub generacji")
    
    
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
