import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


alphabet = { ' ': 0,'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
            'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52
}

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
                x, y = map(float, pair.split(','))
                points.append([x, y, 0])  # Długopis pisze

            # Dodaj stan "w powietrzu" po zakończeniu polyline
            if points:
                last_point = points[-1][:2]  # Pobierz tylko x, y
                points.append([last_point[0], last_point[1], 1])  # Długopis w powietrzu

            polylines.extend(points)


    polylines = np.array(polylines)


    # polylines = align(polylines)

    polylines = coords_to_offsets(polylines)


    # polylines = normalize(polylines)
    # print(polylines)


    return list(polylines)


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
        self.realData = []

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

            text_values = []
            end_prob = []


            for i in range(len(polylines)):
                text_values.append(getTextValue(text, i, len(polylines)))

                end_prob.append(i/len(polylines))


            # Dodanie całej sekwencji z pliku oraz odpowiadającego tekstu
            self.data.append((polylines, text, text_values, end_prob))


        for i in range(len(self.data)):
            polyline = self.data[i][0]
            padded_polyline = self.pad_sequence(polyline, self.max_timesteps)
            input_seq = torch.tensor(padded_polyline[:-1], dtype=torch.float32)
            target_seq = torch.tensor(padded_polyline[1:], dtype=torch.float32)

            padded_alphabet = self.pad_alphabet(self.data[i][2], self.max_timesteps)
            padded_alphabet = torch.tensor(padded_alphabet, dtype=torch.float32)

            input_alphabet = torch.tensor(padded_alphabet[:-1], dtype=torch.float32)
            target_alphabet = torch.tensor(padded_alphabet[1:], dtype=torch.float32)

            padded_end = self.pad_end_probability(self.data[i][3], self.max_timesteps)
            padded_end = torch.tensor(padded_end, dtype=torch.float32)

            padded_end = padded_end.unsqueeze(1)

            input_end = torch.tensor(padded_end[:-1], dtype=torch.float32)
            target_end = torch.tensor(padded_end[1:], dtype=torch.float32)

            full_input_sq = torch.cat((input_seq,input_alphabet,input_end), dim=1)
            full_target_sq = torch.cat((target_seq,target_alphabet,target_end), dim=1)

            self.realData.append((full_input_sq,full_target_sq))

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
        # polyline = self.data[idx][0]
        # padded_polyline = self.pad_sequence(polyline, self.max_timesteps)
        # input_seq = torch.tensor(padded_polyline[:-1], dtype=torch.float32)
        # target_seq = torch.tensor(padded_polyline[1:], dtype=torch.float32)

        # padded_alphabet = self.pad_alphabet(self.data[idx][2], self.max_timesteps)
        # padded_alphabet = torch.tensor(padded_alphabet, dtype=torch.float32)

        # input_alphabet = torch.tensor(padded_alphabet[:-1], dtype=torch.float32)
        # target_alphabet = torch.tensor(padded_alphabet[1:], dtype=torch.float32)

        # padded_end = self.pad_end_probability(self.data[idx][3], self.max_timesteps)
        # padded_end = torch.tensor(padded_end, dtype=torch.float32)

        # padded_end = padded_end.unsqueeze(1)

        # input_end = torch.tensor(padded_end[:-1], dtype=torch.float32)
        # target_end = torch.tensor(padded_end[1:], dtype=torch.float32)

        # print(padded_end)
        # print(input_end)
        # print(target_end)

        # print(padded_alphabet)
        # print(input_alphabet)
        # print(target_alphabet)
        # padded_alphabet = padded_alphabet.unsqueeze(0)
        # print(padded_alphabet.shape)
        # print(input_seq.shape)
        # print(input_alphabet.shape)
        # print(input_end.shape)
        # print(torch.cat((input_seq,input_alphabet,input_end),dim=1).shape)
        # full_input_sq = torch.cat((input_seq,input_alphabet,input_end), dim=1)
        # full_target_sq = torch.cat((target_seq,target_alphabet,target_end), dim=1)

        return self.realData[idx][0], self.realData[idx][1]


def getTextValue(text, timeStep, length):
    valueMap = np.zeros(len(alphabet))

    I = len(text)

    for i, c in enumerate(text):
        idx = alphabet[c]
        # print(i,idx)

        licznik = i/I * length - timeStep
        mianownik = 1/I * length

        value = 1 - np.abs(licznik/mianownik)

        value = max(0,value)
        # print(value)

        if value > valueMap[idx]:
            valueMap[idx] = value

    # print(valueMap)
    return valueMap

def handwriting_collate_fn(batch):
    """
    Funkcja collate do DataLoadera.
    Grupuje dane w batch i wyrównuje ich długości za pomocą paddingu.
    """
    input_seqs, target_seqs = zip(*batch)
    input_seqs = torch.stack(input_seqs)  # Batch input sequences
    target_seqs = torch.stack(target_seqs)  # Batch target sequences
    return input_seqs, target_seqs

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

# folder_path = "output"
# svg_files = ["output/" + file for file in os.listdir(folder_path) if file.endswith('.svg')]
# dataset = HandwritingDataset(svg_files, "output/files.txt")
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=handwriting_collate_fn)

# # getTextValue('witamZ Z',400,500)
# for input_seq, target_seq, in dataloader:
#     print(input_seq.shape)
#     print(target_seq.shape)

