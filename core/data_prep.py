import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.nn.utils.rnn import pad_sequence

'''

        Format danych

        [Used for preprocessing]
        self.data[a][0] - sequence
        self.data[a][1] - text

        [Real use]
        self.realData[a][0] - input sequence
        self.realData[a][1] - target sequence
        self.realData[a][2] - text

'''
# Variable for holding max length of sequence for padding purpose
maximal = 0

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
            i = 0
            for pair in points_str.split():

                x, y = map(float, pair.split(','))
                points.append([x, y, 0])  # Długopis pisze
                i += 1

            # Dodaj stan "w powietrzu" po zakończeniu polyline
            if points:
                last_point = points[-1][:2]  # Pobierz tylko x, y
                points.append([last_point[0], last_point[1], 1])  # Długopis w powietrzu

            polylines.extend(points)


    polylines = adaptive_resample(polylines)

    global maximal
    if maximal < len(polylines):
        maximal = len(polylines)
        print("Max: ",maximal)

    polylines = np.array(polylines)
    
    polylines = coords_to_offsets(polylines)

    return list(polylines)

def adaptive_resample(stroke_data, min_distance=4.0):
    """
    Resample stroke data to reduce resolution while preserving character.
    
    Args:
        stroke_data: Original high-resolution stroke data (x, y, pen_state)
        min_distance: Minimum Euclidean distance between consecutive points
        
    Returns:
        Resampled stroke data with reduced point density
    """
    resampled = [stroke_data[0]]  # Always keep the first point
    last_point = stroke_data[0]
    
    for point in stroke_data[1:]:
        # Always keep pen-up events regardless of distance
        if point[2] != last_point[2]:
            resampled.append(point)
            last_point = point
            continue
            
        # Calculate Euclidean distance
        distance = ((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)**0.5
        
        if distance >= min_distance:
            resampled.append(point)
            last_point = point
    
    return resampled


'''
    Format danych:
    1 - wymiar : plik czyli np. [0] - oznacza pierwszy plik
    2- wymiar punkt czyli np. [2][4] - oznacza 3 plik 5 punkt
'''
class HandwritingDataset(Dataset):
    def __init__(self, svg_files, text_files):
        """
        Dataset dla uczenia modelu na danych ręcznego pisma w formacie SVG.
        Args:
            svg_files (list): Lista ścieżek do plików SVG.
            text_file (str): Ścieżka do pliku tekstowego zawierającego teksty (jedna linia na plik SVG).
        """
        self.data = []  # Lista sekwencji (każda sekwencja to lista punktów)
        # self.texts = []  # Lista tekstów odpowiadających danym
        self.max_timesteps = 720
        self.realData = []

        all_texts = []


        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='windows-1252') as f:
                    lines = f.readlines()
                    texts = [line.strip() for line in lines]
                    all_texts.extend(texts)
            except Exception as e:
                print(f"Błąd wczytywania pliku {text_file}: {e}")
        
        # Sprawdzenie, czy liczba tekstów zgadza się z liczbą plików SVG
        if len(all_texts) != len(svg_files):
            raise ValueError("Liczba tekstów w plikach nie zgadza się z liczbą plików SVG.")
        
        # Wczytanie danych z plików SVG
        data_counter = 0
        for file, text in zip(svg_files, all_texts):
            # Parsowanie pliku SVG na punkty
            polylines = parse_svg(file)
            
            # Dodanie całej sekwencji z pliku oraz odpowiadającego tekstu
            self.data.append((polylines, text))

            data_counter += 1

            # if data_counter % 500 == 0:
            #     break
        
        for i in range(len(self.data)):
            polyline = self.data[i][0]
            padded_polyline = self.pad_sequence(polyline, self.max_timesteps)
            input_seq = torch.tensor(padded_polyline[:-1], dtype=torch.float32)
            target_seq = torch.tensor(padded_polyline[1:], dtype=torch.float32)            

            self.realData.append((input_seq,target_seq,self.data[i][1]))
            # print(self.realData[-1])

            if i % 100 == 0:
                print(str(i) + "/" + str(len(self.data)) + " prepared")

        print("Data ready")

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
        print(len(self.realData))
        return len(self.realData)

    def pad_sequence(self, sequence, max_length):
        """
        Uzupełnia sekwencję zerami do określonej maksymalnej długości.
        """
        sequence_length = len(sequence)
        if sequence_length < max_length:
            padding = [[0, 0, 0]] * (max_length - sequence_length)  # Dodaj zerowe timestepy
            sequence.extend(padding)
        return sequence[:max_length]  # Przytnij do max_length (dla bezpieczeństwa)

    def pad_alphabet(self, sequence, max_length):
        sequence_length = len(sequence)
        element_size = len(sequence[0]) if sequence else 0
        if sequence_length < max_length:
            padding = [[0] * element_size] * (max_length - sequence_length)  # Dodaj zerowe timestepy
            sequence.extend(padding)
        return sequence[:max_length]  # Przytnij do max_length (dla bezpieczeństwa)

    def pad_end_probability(self, sequence, max_length):
        sequence_length = len(sequence)
        if sequence_length < max_length:
            padding = [0] * (max_length - sequence_length)  # Dodaj zerowe timestepy
            sequence.extend(padding)
        return sequence[:max_length]  # Przytnij do max_length (dla bezpieczeństwa)

    def __getitem__(self, idx):
        return self.realData[idx][0], self.realData[idx][1], self.realData[idx][2]

def handwriting_collate_fn(batch):
    """
    Collate function for DataLoader.
    Groups data into a batch and pads variable-length sequences.
    """
    input_seqs, target_seqs, texts = zip(*batch)

    # Pad the input sequences (assumed to be tensors of shape [seq_len, input_dim])
    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=0)

    # Pad the target sequences (assumed to be tensors of shape [seq_len, target_dim])
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=0)

    # Pad the text sequences (if they are also variable-length, e.g. sequences of character indices or embeddings)
    # texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return input_seqs, target_seqs, texts

def align(coords):
    """
    Corrects for global slant/offset in handwriting strokes using NumPy.
    """
    coords = np.copy(coords)  # Tworzy kopię danych wejściowych

    # Oddzielne kolumny X i Y
    X = coords[:, 0].reshape(-1, 1)
    Y = coords[:, 1].reshape(-1, 1)

    # Tworzenie macierzy X z kolumną jedynek
    ones = np.ones((X.shape[0], 1))
    X = np.hstack([ones, X])

    # Obliczanie współczynników offset i slope
    XtX = X.T @ X
    XtY = X.T @ Y
    coeffs = np.linalg.solve(XtX, XtY).squeeze()
    offset, slope = coeffs[0], coeffs[1]

    # Obliczanie kąta i macierzy rotacji
    theta = np.arctan(slope)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotacja współrzędnych i korekta offsetu
    coords[:, :2] = coords[:, :2] @ rotation_matrix - offset

    return coords

def normalize(offsets):
    """
    Normalizes strokes to median unit norm using NumPy.
    """
    offsets = np.copy(offsets)  # Tworzy kopię danych wejściowych

    # Obliczanie mediany normy
    norms = np.linalg.norm(offsets[:, :2], axis=1)
    median_norm = np.median(norms)

    # Normalizacja offsetów
    offsets[:, :2] /= median_norm

    return offsets

'''
    Zmienia punkty w offsety i wtedy
    Dodaje 0 0 1 na start

    czyli tak jakby zaczynamy od 0,0 i wtedy od razu offsety pomiędzy
    czyli nie ma różnicy gdzie się zacznie pisać
'''
def coords_to_offsets(coords):
    """
    convert from coordinates to offsets
    """
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    offsets = np.concatenate([np.array([[0, 0, 1]]), offsets], axis=0)
    return offsets