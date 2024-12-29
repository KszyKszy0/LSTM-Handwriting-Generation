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
folder_path = "output"

# Lista nazw plików z rozszerzeniem .svg
svg_files = ["output/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]

print(svg_files)


epochs = params.epochs  # Liczba epok treningowych
learning_rate = params.learning_rate  # Szybkość uczenia
hidden_size = params.hidden_size  # Rozmiar warstwy ukrytej
num_layers = params.num_layers  # Liczba warstw LSTM
num_mixtures = params.num_mixtures  # Liczba komponentów mieszanki

model = rnn.MixtureDensityNetwork(input_size=3, hidden_size=hidden_size, num_layers=num_layers, num_mixtures=num_mixtures)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# svg_files = ['output/00001.svg','output/00002.svg']  # Podmień na rzeczywiste ścieżki
dataset = data.HandwritingDataset(svg_files)
dataloader = data.DataLoader(dataset, batch_size=1)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_seq, target_seq in dataloader:
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
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoka [{epoch + 1}/{epochs}], Strata: {total_loss:.4f}")
    torch.save(model.state_dict(), f"models/mdn_epoch_{epoch + 1}.pth")