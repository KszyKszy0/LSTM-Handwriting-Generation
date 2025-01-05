import string
import new_model
import torch
import torch.optim as optim
import os
import core.data_prep as data

# dict(zip(string.ascii_lowercase, range(1,27)))
# print(dict(zip(string.ascii_uppercase, range(27,27+26))))




INPUT_SIZE = 3+len(data.alphabet)
HIDDEN_SIZE = 40
OUTPUT_SIZE = 3

model = new_model.model(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE)

EPOCHS = 10_000
LEARNING_RATE = 1e-6

optimizer = optim.Adam(model.parameters, LEARNING_RATE)

# Ścieżka do folderu
folder_path = "output"

files_content = "output/files.txt"

# Lista nazw plików z rozszerzeniem .svg
svg_files = ["output/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]

dataset = data.HandwritingDataset(svg_files, files_content)
dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=data.handwriting_collate_fn)