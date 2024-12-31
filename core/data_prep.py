import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

'''

        Format danych
        self.data[a][b]

        self.data[a][0][0-1]
        self.data[a][1]

        [a][0][0] - input, [a][0][1] - target

        [a][1] - tekst

'''
# Funkcja do wczytania i sparsowania pliku SVG
def parse_svg(file_path):
    """
    Funkcja bierze plik SVG, przetwarza wszystkie polyline i tworzy tablicę punktów (x, y, 0-1),
    gdzie 0 oznacza, że długopis jest w powietrzu, a 1, że pisze.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    polylines = []

    x = 0
    y = 0
    # Znajdź wszystkie polyline w SVG
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points_str = polyline.attrib.get('points', '').strip()
        if points_str:
            # Zamień punkty na listę par współrzędnych
            points = []
            for pair in points_str.split():
                previous_x = x
                previous_y = y
                x, y = map(float, pair.split(','))
                points.append([x - previous_x, y - previous_y, 0])  # Długopis pisze

            # Dodaj stan "w powietrzu" po zakończeniu polyline
            if points:
                last_point = points[-1][:2]  # Pobierz tylko x, y
                points.append([last_point[0], last_point[1], 1])  # Długopis w powietrzu

            polylines.extend(points)
    return polylines


'''
    Format danych:
    1 - wymiar : plik czyli np. [0] - oznacza pierwszy plik
    2- wymiar punkt czyli np. [2][4] - oznacza 3 plik 5 punkt
'''
class HandwritingDataset(Dataset):
    def __init__(self, svg_files, text_file):
        """
        Dataset dla uczenia modelu na danych ręcznego pisma w formacie SVG.
        Args:
            svg_files (list): Lista ścieżek do plików SVG.
            text_file (str): Ścieżka do pliku tekstowego zawierającego teksty (jedna linia na plik SVG).
        """
        self.data = []  # Lista sekwencji (każda sekwencja to lista punktów)
        self.texts = []  # Lista tekstów odpowiadających danym
        self.max_timesteps = 2000

        # Wczytanie tekstów z pliku
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.texts = [line.strip() for line in lines]

        # Sprawdzenie, czy liczba tekstów zgadza się z liczbą plików SVG
        if len(self.texts) != len(svg_files):
            raise ValueError("Liczba tekstów w pliku nie zgadza się z liczbą plików SVG.")

        # Wczytanie danych z plików SVG
        for file, text in zip(svg_files, self.texts):
            # Parsowanie pliku SVG na punkty
            polylines = parse_svg(file)
            # Dodanie całej sekwencji z pliku oraz odpowiadającego tekstu
            self.data.append((polylines, text))

        # Normalizacja danych
        self.normalize_data()


    def normalize_data(self):
        """
        Normalizacja współrzędnych x, y w danych SVG. Flaga (0-1) pozostaje bez zmian.
        Tekst (druga kolumna krotek) nie jest modyfikowany.
        """
        # Ekstrakcja wszystkich punktów x, y do jednego arraya
        all_points = np.concatenate(
            [np.array(seq[0])[:, :2] for seq in self.data], axis=0  # seq[0] to `polylines`
        )
        self.mean = np.mean(all_points, axis=0)
        self.std = np.std(all_points, axis=0)

        # Zapis średniej i odchylenia standardowego do pliku
        with open('norm.txt', 'w') as file:
            np.savetxt(file, np.column_stack((self.mean, self.std)))

        # Normalizacja x, y w `polylines`
        for i in range(len(self.data)):
            polylines, text = self.data[i]
            normalized_polylines = [
                [
                    (point[0] - self.mean[0]) / self.std[0],  # Normalizacja x
                    (point[1] - self.mean[1]) / self.std[1],  # Normalizacja y
                    point[2],  # Flaga bez zmian
                ]
                for point in polylines
            ]
            # Aktualizacja znormalizowanych danych, pozostawiając tekst bez zmian
            self.data[i] = (normalized_polylines, text)

    def __len__(self):
        """
        Zwraca liczbę plików (sekwencji) w zbiorze danych.
        """
        return len(self.data)

    def pad_sequence(self, sequence, max_length):
        """
        Uzupełnia sekwencję zerami do określonej maksymalnej długości.
        """
        sequence_length = len(sequence)
        if sequence_length < max_length:
            padding = [[0, 0, 0]] * (max_length - sequence_length)  # Dodaj zerowe timestepy
            sequence.extend(padding)
        return sequence[:max_length]  # Przytnij do max_length (dla bezpieczeństwa)

    def __getitem__(self, idx):
        polyline = self.data[idx][0]
        padded_polyline = self.pad_sequence(polyline, self.max_timesteps)
        input_seq = torch.tensor(padded_polyline[:-1], dtype=torch.float32)
        target_seq = torch.tensor(padded_polyline[1:], dtype=torch.float32)
        return input_seq, target_seq, self.data[idx][1]




def handwriting_collate_fn(batch):
    """
    Funkcja collate do DataLoadera.
    Grupuje dane w batch i wyrównuje ich długości za pomocą paddingu.
    """
    input_seqs, target_seqs, texts = zip(*batch)
    input_seqs = torch.stack(input_seqs)  # Batch input sequences
    target_seqs = torch.stack(target_seqs)  # Batch target sequences
    return input_seqs, target_seqs, texts

# folder_path = "output"
# svg_files = ["output/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]
# dataset = HandwritingDataset(svg_files, "output/files.txt")
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=handwriting_collate_fn)


# for input_seq, target_seq, text in dataloader:
#     print(input_seq.shape)
#     print(target_seq.shape)
#     print(text)