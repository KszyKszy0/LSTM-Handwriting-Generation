import torch
import torch.nn as nn
import core.data_prep as data
import new_model as model_file
import xml.etree.ElementTree as ET
from xml.dom import minidom
import core.rnn as rnn_file
import core.params as params

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
        polyline_points.append(f"{x+50},{y+50}")
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

def sample_from_mixture(mixture_weights, means, std_devs, correlations):
    """
    Samples a point (dx, dy) from the predicted Gaussian mixture model.
    """
    num_mixtures = mixture_weights.shape[-1]

    # Choose a mixture component
    mixture_idx = torch.multinomial(mixture_weights, 1).squeeze()

    # Extract parameters for chosen component
    mean = means[..., mixture_idx, :]
    std = std_devs[..., mixture_idx, :]
    corr = correlations[..., mixture_idx]

    # Sample from bivariate Gaussian
    mean_x, mean_y = mean[..., 0], mean[..., 1]
    std_x, std_y = std[..., 0], std[..., 1]

    # Compute covariance matrix
    cov_xy = corr * std_x * std_y
    cov_matrix = torch.stack([
        torch.stack([std_x**2, cov_xy], dim=-1),
        torch.stack([cov_xy, std_y**2], dim=-1)
    ], dim=-2)

    # Sample from multivariate normal distribution
    mvn_samples = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix).sample()
    return mvn_samples[..., 0], mvn_samples[..., 1]

def create_new_svg(points, output_file):
    """
    Tworzy plik SVG z listy punktów, oddzielając linie (stroke) na podstawie end_of_stroke.

    Args:
        points: Lista punktów [(x, y, eos)].
        output_file: Ścieżka do pliku wynikowego SVG.
    """
    # Tworzymy główny element SVG
    root = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", version="1.1")

    # Lista do przechowywania punktów dla bieżącej linii (stroke)
    polyline_points = []

    # Iterujemy po punktach
    for x, y, eos in points:
        # print(x,y,eos)
        # Dodajemy punkt do bieżącej linii
        polyline_points.append(f"{x+50},{y+50}")

        # print(polyline_points)
        # Jeżeli eos == 1, kończymy bieżącą linię i zaczynamy nową
        if eos == 1:
            # Dodajemy polyline do SVG
            polyline = ET.SubElement(root, "polyline", points=" ".join(polyline_points),
                                     style="fill:none;stroke:black;stroke-width:1")
            # Resetujemy listę punktów dla nowej linii
            polyline_points = []

    # Jeżeli są punkty w polyline_points, dodajemy ostatnią linię
    if polyline_points:
        ET.SubElement(root, "polyline", points=" ".join(polyline_points),
                      style="fill:none;stroke:black;stroke-width:1")

    # Tworzymy wynikowy plik SVG
    tree = ET.ElementTree(root)
    tree.write(output_file)

    print(f"SVG saved as {output_file}")

def get_seq_auto(start, text, limit, model):
    model.eval()
    hid = model.init_hidden(1)

    start = torch.tensor(start)

    current_point = start

    all_points = []
    all_points.append((current_point[0].item(),current_point[1].item(),1))

    start_state = torch.tensor([0,0,1],dtype=torch.float32)
    alph_start = torch.tensor(data.getTextValue(text, 1, limit),dtype=torch.float32)
    end_start = torch.tensor([0],dtype=torch.float32)

    input_state = torch.concat((start_state,alph_start,end_start)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for step in range(limit):
            eos, mixture_weights, means, std_devs, correlations, hid = model(input_state, hid)

            dx, dy = sample_from_mixture(mixture_weights.squeeze(0), means.squeeze(0), std_devs.squeeze(0), correlations.squeeze(0))

            all_points.append((dx.item(),dy.item(),eos))

            pen_state = 1 if torch.rand(1).item() > eos.item() else 0

            input_state = torch.cat((torch.tensor([dx, dy, pen_state], dtype=torch.float32), alph_start, end_start)).unsqueeze(0).unsqueeze(0)

    # Zamiana deltas na współrzędne absolutne
    absolute_points = []

    x, y = 0.0, 0.0
    #denormalizacja żeby poprawnie dodać współrzędne
    for i in range(len(all_points)):
        x += all_points[i][0]
        y += all_points[i][1]
        absolute_points.append((x, y, all_points[i][2]))

    # Zapis do SVG
    create_new_svg(absolute_points, "test.svg")
    print(f"Sequence saved")

MODEL_PATH = "models/mdn_epoch_44 2.4820514917373657.pth"

state_dict = torch.load(MODEL_PATH, weights_only=True)
# drawer = model_file.model(57,800,57)
model_rnn = rnn_file.MixtureDensityNetwork(input_size=57, hidden_size=params.hidden_size, num_layers=params.num_layers, num_mixtures=params.num_mixtures)
model_rnn.load_state_dict((state_dict['model_state_dict']))

get_seq_auto([0,0,1], "w", 50, model_rnn)
