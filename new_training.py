import string
import new_model
import torch
import torch.optim as optim
import os
import core.data_prep as data
import torch.nn as nn
import datetime

# dict(zip(string.ascii_lowercase, range(1,27)))
# print(dict(zip(string.ascii_uppercase, range(27,27+26))))


MODEL_PATH = "correct/"

INPUT_SIZE = 4+len(data.alphabet)
HIDDEN_SIZE = 800
OUTPUT_SIZE = 4+len(data.alphabet)

model = new_model.model(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,1)

EPOCHS = 100_000
LEARNING_RATE = 1e-6

optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

# Ścieżka do folderu
folder_path = "output"

files_content = "output/files.txt"

# Lista nazw plików z rozszerzeniem .svg
svg_files = ["output/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]

dataset = data.HandwritingDataset(svg_files, files_content)
dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=data.handwriting_collate_fn)

cords_loss = nn.L1Loss()

# alphabet_loss = nn.CrossEntropyLoss()

end_loss = nn.BCELoss()

eos_loss = nn.BCELoss()

# model_to_load = "new_model_arch/315 110.70211815834045"
# checkpoint = torch.load(model_to_load, weights_only=True)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def custom_loss(preds, target_probs):
    """
    preds: Wyjścia modelu [batch_size, num_classes]
    target_probs: Docelowe wartości (mogą nie sumować się do 1) [batch_size, num_classes]
    """
    # Dodanie małej wartości, by uniknąć log(0)
    preds = torch.sigmoid(preds)  # Przekształcamy na przedział [0, 1] (jeśli wymagane)
    log_preds = torch.log(preds + 1e-9)

    # Użycie maski, by obliczać stratę tylko dla aktywnych elementów
    loss = -torch.sum(target_probs * log_preds, dim=-1)  # Punktowe porównanie
    return loss.mean()

for i in range(EPOCHS):
    model.train()
    total = 0

    total_mae = 0
    total_ce = 0
    total_bce = 0
    total_eos = 0

    for input_seq, target_seq in dataloader:

        sample_size = input_seq.shape[0]

        hidden = model.init_hidden(sample_size)

        points, eos, alph, end, hidden = model(input_seq, hidden)

        batch_mae = cords_loss(points,target_seq[:,:,:2])

        batch_ce = custom_loss(alph,target_seq[:,:,3:56])

        batch_bce = end_loss(end,target_seq[:,:,56])

        batch_eos = eos_loss(eos, target_seq[:,:,2])

        optimizer.zero_grad()

        total_batch_loss = batch_mae + batch_ce + batch_bce + batch_eos

        total_batch_loss.backward()

        optimizer.step()

        total += total_batch_loss.item()

        total_mae += batch_mae.item()
        total_ce += batch_ce.item()
        total_bce += batch_bce.item()
        total_eos += batch_eos.item()

    print(f"Epoka [{i}/{EPOCHS}], Loss: {total:.4f}")
    print(f"  MAE: {total_mae:.4f}, Alphabet: {total_ce:.4f}, BCE (End): {total_bce:.4f}, BCE (EOS): {total_eos:.4f}")
    with open("logs/" + "log.txt", "a") as file:
        file.write(f"Epoka [{i}/{EPOCHS}], Loss: {total:.4f}")
        file.write(f"  MAE: {total_mae:.4f}, Alphabet: {total_ce:.4f}, BCE (End): {total_bce:.4f}, BCE (EOS): {total_eos:.4f}" + "\n")
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH + str(i) + " " + str(total))