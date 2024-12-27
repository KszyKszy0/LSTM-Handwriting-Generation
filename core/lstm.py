import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn

# Funkcja do wczytania i sparsowania pliku SVG
def parse_svg(file_path):
    """
    Funkcja bierze plik SVG, przetwarza wszystkie polyline i tworzy tablicę punktów (x, y, 0-1),
    gdzie 0 oznacza, że długopis jest w powietrzu, a 1, że pisze.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    polylines = []

    # Znajdź wszystkie polyline w SVG
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points_str = polyline.attrib.get('points', '').strip()
        if points_str:
            # Zamień punkty na listę par współrzędnych
            points = []
            for pair in points_str.split():
                x, y = map(float, pair.split(','))
                points.append([x, y, 1])  # Długopis pisze

            # Dodaj stan "w powietrzu" po zakończeniu polyline
            if points:
                last_point = points[-1][:2]  # Pobierz tylko x, y
                points.append([last_point[0], last_point[1], 0])  # Długopis w powietrzu

            polylines.extend(points)
    return polylines


'''
    Format danych:
    1 - wymiar : plik czyli np. [0] - oznacza pierwszy plik
    2- wymiar punkt czyli np. [2][4] - oznacza 3 plik 5 punkt
'''
class HandwritingDataset(Dataset):
    def __init__(self, svg_files):
        """
        Dataset dla uczenia modelu na danych ręcznego pisma w formacie SVG.
        """
        self.data = []  # Lista sekwencji (każda sekwencja to lista punktów)
        for file in svg_files:
            # Parsowanie pliku SVG na punkty
            polylines = parse_svg(file)
            self.data.append(polylines)  # Dodanie całej sekwencji z pliku

        # Normalizacja danych
        self.normalize_data()


    def normalize_data(self):
        """
        Normalizacja współrzędnych x, y. Flaga (0-1) nie jest normalizowana.
        """
        all_points = np.concatenate([np.array(seq)[:, :2] for seq in self.data], axis=0)  # Ekstrakcja x, y
        self.mean = np.mean(all_points, axis=0)
        self.std = np.std(all_points, axis=0)

        # Normalizacja x, y; flaga (0-1) pozostaje bez zmian
        for i in range(len(self.data)):
            self.data[i] = [
                [(point[0] - self.mean[0]) / self.std[0],
                 (point[1] - self.mean[1]) / self.std[1],
                 point[2]]  # Flaga niezmieniona
                for point in self.data[i]
            ]

    def __len__(self):
        """
        Zwraca liczbę plików (sekwencji) w zbiorze danych.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Zwraca sekwencję wejściową i docelową dla danego indeksu pliku.
        """
        polyline = self.data[idx]
        input_seq = torch.tensor(polyline[:-1], dtype=torch.float32)  # Wszystko oprócz ostatniego punktu
        target_seq = torch.tensor(polyline[1:], dtype=torch.float32)  # Wszystko oprócz pierwszego punktu
        return input_seq, target_seq


class HandwritingLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=3):
        super(HandwritingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass przez LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        output = self.fc(lstm_out)  # output: [batch_size, seq_len, output_size]
        return output


svg_files = ['output/00001.svg']  # Podmień na rzeczywiste ścieżki
dataset = HandwritingDataset(svg_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# poly = parse_svg('output/00001.svg')
# print(poly)

model = HandwritingLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Pętla uczenia
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()

        # Forward pass
        output = model(input_seq)

        # Obliczenie straty
        loss = loss_fn(output, target_seq)
        epoch_loss += loss.item()

        # Backward pass i optymalizacja
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")


def generate_sequence(model, start_point, seq_len=100):
    """
    Generuje sekwencję punktów za pomocą wyuczonego modelu.
    """
    model.eval()
    generated_points = [start_point]
    current_input = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


    for _ in range(seq_len):
        with torch.no_grad():
            # Przewidywanie kolejnego punktu
            next_point = model(current_input)
            # print(next_point)

            next_point = next_point.squeeze(0).squeeze(0).numpy()
            generated_points.append(next_point)

            # Ustawienie nowego wejścia
            current_input = torch.tensor(next_point, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return generated_points

def save_to_svg(polylines, output_file):
    """
    Zapisuje wygenerowane polilinie do pliku SVG.
    """
    svg_root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        "version": "1.1",
        "baseProfile": "full",
        "width": "500",
        "height": "500"
    })

    for polyline in polylines:
        points_str = " ".join(f"{x},{y}" for x, y, flag in polyline if flag >= 0.5)
        polyline_elem = ET.SubElement(svg_root, "polyline", {
            "fill": "none",
            "stroke": "black",
            "stroke-width": "2",
            "points": points_str
        })

    tree = ET.ElementTree(svg_root)
    tree.write(output_file)

# Generowanie nowej sekwencji
start_point = [34.0, 47.0, 1.0]  # Początkowy punkt
generated_sequence = generate_sequence(model, start_point, seq_len=200)



# Przetworzenie na polilinie (na przykład po 50 punktów)
polylines = [generated_sequence[i:i + 50] for i in range(0, len(generated_sequence), 50)]

print(polylines)

# Zapis do SVG
save_to_svg(polylines, "generated_output.svg")
print("Wygenerowano plik SVG: generated_output.svg")
