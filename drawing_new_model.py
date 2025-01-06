import torch
import torch.nn as nn
import core.data_prep as data
import new_model as model_file
import xml.etree.ElementTree as ET
from xml.dom import minidom

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

def get_seq_auto(start, text, limit, model):
    model.eval()
    hid = model.init_hidden(1)

    start = torch.tensor(start)

    current_point = start

    all_points = []
    all_points.append(current_point)

    start_state = torch.tensor([0,0,1],dtype=torch.float32)
    alph_start = torch.tensor(data.getTextValue(text, 1, limit),dtype=torch.float32)
    end_start = torch.tensor([0],dtype=torch.float32)

    input_state = torch.concat((start_state,alph_start,end_start)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        for step in range(limit):
            coords, eos, alph, end, hid = model(input_state, hid)

            eos = eos.unsqueeze(0)
            end = end.unsqueeze(0)

            all_points.append((coords[0,0,0].item(),coords[0,0,1].item(),eos[0,0,0].item()))

            input_state = torch.concat((coords,eos,alph,end),dim=2)


    # Zamiana deltas na współrzędne absolutne
    absolute_points = []

    # print(all_points)

    x, y = 0.0, 0.0
    #denormalizacja żeby poprawnie dodać współrzędne
    for i in range(len(all_points)):
        x += all_points[i][0]
        y += all_points[i][1]
        absolute_points.append((x, y, all_points[i][2]))

    # Zapis do SVG
    create_svg(absolute_points, "test.svg")
    print(f"Sequence saved")

MODEL_PATH = "correct/9 125.68188202381134"

state_dict = torch.load(MODEL_PATH, weights_only=True)
drawer = model_file.model(57,800,57)
drawer.load_state_dict(state_dict['model_state_dict'])

get_seq_auto([0,0,1], "test", 2000, drawer)
