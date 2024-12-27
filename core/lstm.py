import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Funkcja do wczytania i sparsowania pliku SVG
def parse_svg(file_path):
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
                points.append([x, y, 1])
            polylines.append(points)
            points.append([x, y, 0])
    return polylines

poly = parse_svg('output/00001.svg')
print(poly)