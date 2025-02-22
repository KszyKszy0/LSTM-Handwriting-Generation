import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rnn
import data_prep as data
import torch.optim as optim
import os
import params

# Ścieżka do folderu
folder_path = "single_letters"

files_content = "single_letters/files.txt"

# Lista nazw plików z rozszerzeniem .svg
svg_files = ["single_letters/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]

print(svg_files)

MODEL_PATH = "models/mdn_epoch_"

epochs = params.epochs  # Liczba epok treningowych
learning_rate = params.learning_rate  # Szybkość uczenia
hidden_size = params.hidden_size  # Rozmiar warstwy ukrytej
num_layers = params.num_layers  # Liczba warstw LSTM
num_mixtures = params.num_mixtures  # Liczba komponentów mieszanki

model = rnn.MixtureDensityNetwork(input_size=57, hidden_size=hidden_size, num_layers=num_layers, num_mixtures=num_mixtures)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# state_dict = torch.load("models\mdn_epoch_.pth1 201.20002016425133", weights_only=True)
# model.load_state_dict(state_dict['model_state_dict'])
# optimizer.load_state_dict(state_dict['optimizer_state_dict'])

# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=params.min_learning_rate)

# svg_files = ['output/00001.svg','output/00002.svg']  # Podmień na rzeczywiste ścieżki
dataset = data.HandwritingDataset(svg_files, files_content)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=data.handwriting_collate_fn)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (input_seq, target_seq) in enumerate(dataloader):
        batch_size, seq_len, _ = input_seq.size()

        # Inicjalizacja ukrytego stanu LSTM
        hidden = model.init_hidden(batch_size)

        # Forward pass
        end_of_stroke, mixture_weights, means, std_devs, correlations, hidden = model(input_seq, hidden)

        # Oblicz stratę
        loss = rnn.mdn_loss(end_of_stroke, mixture_weights, means, std_devs, correlations, target_seq)

        # Backward pass i optymalizacja
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        print(f"Batch [{i + 1}/{len(dataloader)}], Strata: {loss.item():.4f}")

    print(f"Epoka [{epoch + 1}/{epochs}], Strata: {total_loss/len(dataloader):.4f}")
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH + str(epoch) + " " + str(total_loss/len(dataloader)) + ".pth")
    # scheduler.step()
    # current_lr = scheduler.get_last_lr()[0]
    # print(f"Learning Rate after Epoch {epoch+1}: {current_lr:.6f}")