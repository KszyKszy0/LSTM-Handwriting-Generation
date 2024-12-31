import torch
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import rnn
import params

def generate_sequence(model, start_point, seq_length=100, num_mixtures=20, output_file="output.svg"):
    """
    Generuje sekwencję punktów na podstawie modelu i zapisuje ją do pliku SVG jako polyline.

    Args:
        model: Wytrenowany model MDN.
        start_point: Punkt początkowy [dx, dy, eos].
        seq_length: Liczba punktów do wygenerowania.
        num_mixtures: Liczba komponentów mieszanki w modelu.
        output_file: Ścieżka do pliku wynikowego SVG.
    """
    # data = np.loadtxt('norm.txt')
    # # Rozdzielenie średnich i odchyleń standardowych
    # meanf = data[:, 0]  # Pierwsza kolumna
    # stdf = data[:, 1]   # Druga kolumna
    # start_point = [
    #         (start_point[0] - meanf[0]) / stdf[0],
    #         (start_point[1] - meanf[1]) / stdf[1],
    #         start_point[2]
    #     ]

    model.eval()
    hidden = model.init_hidden(1)  # Inicjalizacja stanu ukrytego
    start_point = torch.tensor(start_point, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 3]

    generated_points = []


    # print(meanf)
    # print(stdf)


    current_point = start_point

    with torch.no_grad():
        for _ in range(seq_length):
            # Przewidzenie parametrów dla bieżącego punktu
            eos, weights, means, std_devs, correlations, hidden = model(current_point, hidden)

            # Rozkład mieszanki Gaussowskiej
            mixture_idx = torch.multinomial(weights.squeeze(0).squeeze(0), 1).item()

            # Wyciągnięcie parametrów dla wybranego komponentu mieszanki
            mean = means[0, 0, mixture_idx]  # [2]
            std_dev = std_devs[0, 0, mixture_idx]  # [2]
            rho = correlations[0, 0, mixture_idx]  # Skalar

            # Losowanie z 2D rozkładu Gaussowskiego
            cov_matrix = torch.tensor([[std_dev[0] ** 2, rho * std_dev[0] * std_dev[1]],
                                        [rho * std_dev[0] * std_dev[1], std_dev[1] ** 2]])
            gaussian_sample = torch.distributions.MultivariateNormal(mean, cov_matrix).sample()

            # Przewidzenie prawdopodobieństwa końca rysowania
            eos_prob = eos.item()
            eos_sample = 1 if np.random.rand() < eos_prob else 0

            # Aktualizacja bieżącego punktu
            dx, dy = gaussian_sample.tolist()

            # Ustawienie jako kolejny input outputu sieci (nwm czy poprawnie znormalizowany)
            current_point = torch.tensor([[dx, dy, eos_sample]], dtype=torch.float32).unsqueeze(0)

            # Dodanie punktu do wygenerowanej sekwencji
            generated_points.append([dx, dy, eos_sample])

    # Zamiana deltas na współrzędne absolutne
    absolute_points = []

    x, y = 0, 0
    #denormalizacja żeby poprawnie dodać współrzędne
    for dx, dy, eos in generated_points:
        x += dx
        y += dy
        absolute_points.append((x, y, eos))


    # Zapis do SVG
    create_svg(absolute_points, output_file)
    print(f"Sequence saved to {output_file}")

def create_svg(points, output_file):
    """
    Tworzy plik SVG z listy punktów.

    Args:
        points: Lista punktów [(x, y, eos)].
        output_file: Ścieżka do pliku wynikowego SVG.
    """
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", version="1.1")

    polyline_points = []
    for x, y, eos in points:
        polyline_points.append(f"{x},{y}")
        if eos == 1:  # Nowa linia
            if polyline_points:
                polyline = ET.SubElement(root, "polyline", points=" ".join(polyline_points),
                                         style="fill:none;stroke:black;stroke-width:1")
                polyline_points = []  # Resetuj punkty dla nowej polyline

    # Dodaj ostatnią linię, jeśli istnieje
    if polyline_points:
        ET.SubElement(root, "polyline", points=" ".join(polyline_points),
                      style="fill:none;stroke:black;stroke-width:1")

    # Formatowanie i zapis do pliku
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_file, "w") as f:
        f.write(reparsed.toprettyxml(indent="  "))

# Przykład użycia
model_path = "models/mdn_epoch_10.pth"  # Ścieżka do zapisanego modelu
model = rnn.MixtureDensityNetwork(input_size=3, hidden_size=params.hidden_size, num_layers=params.num_layers, num_mixtures=params.num_mixtures)
model.load_state_dict(torch.load(model_path))

start_point = [0.0, 0.0, 0.0]  # Punkt początkowy
generate_sequence(model, start_point, seq_length=500, output_file="generated.svg")