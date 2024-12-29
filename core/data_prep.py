import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
            x = 0
            y = 0
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
        # self.normalize_data()
        print(self.data)


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



svg_files = ['output/00001.svg','output/00002.svg']  # Podmień na rzeczywiste ścieżki
dataset = HandwritingDataset(svg_files)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


