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

        Data format

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

# Function for loading and parsing SVGs
def parse_svg(file_path):
    """
    Parses an SVG file, extracting all polyline points as (x, y, pen_state).
    pen_state: 1 = pen up (move), 0 = pen down (draw)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    polylines = []


    for polyline in root.findall('.//svg:polyline', ns):
        points_str = polyline.attrib.get('points', '').strip()

        if points_str:
            points = []
            for pair in points_str.split():
                x, y = map(float, pair.split(','))
                points.append([x, y, 0])  # Pen down

            # Append pen up at the end
            if points:
                last_point = points[-1][:2]
                points.append([last_point[0], last_point[1], 1])  # Pen up

            polylines.extend(points)

    polylines = adaptive_resample(polylines)

    global maximal
    if maximal < len(polylines):
        maximal = len(polylines)
        print("Max: ",maximal)

    polylines = np.array(polylines)
    
    polylines = coords_to_offsets(polylines)

    return list(polylines)

def adaptive_resample(stroke_data, min_distance=2.0):
    """
    Resample stroke data to reduce resolution while preserving character.
    Small and frequent changes do not provide meaningful information for learning purpose.
    Most of the times they only impact length of the sequence.
    It is much easier to learn from short sequences. 
    That's why we use this function to limit the number of points taken into consideration while learning.
    
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
    Data format:
    1 - dimension is a file i.e. [0] - first file
    2 - dimension is a point which means np. [2][4] - 3rd file, 5 point of that file
'''
class HandwritingDataset(Dataset):
    def __init__(self, svg_files, text_files):
        """
        Dataset for model learning handwriting on SVGs
        Args:
            svg_files (list): List of filepaths to all SVG files.
            text_file (str): Filepath to txt file where every line corresponds to every svg file in order 
        """
        self.data = []  # Lista sekwencji (każda sekwencja to lista punktów)
        self.max_timesteps = 1550
        self.realData = []

        all_texts = text_files
        
        # Checking if amount of lines in the text file, and amount of SVG files is the same.
        if len(all_texts) != len(svg_files):
            raise ValueError("Liczba tekstów w plikach nie zgadza się z liczbą plików SVG.")
        
        # Parsing SVG files
        data_counter = 0
        for file, text in zip(svg_files, all_texts):
            # Extracting points
            polylines = parse_svg(file)

            if(len(polylines) > 1500):
                print(text)
                print(len(polylines))
                continue

            if(len(polylines) > len(text) * 45):
                print(text)
                print(len(polylines))
                continue
            
            # Appending whole sequence and corresponding text
            self.data.append((polylines, text))

            data_counter += 1
        
        for i in range(len(self.data)):
            polyline = self.data[i][0]
            padded_polyline = self.pad_sequence(polyline, self.max_timesteps)
            input_seq = torch.tensor(padded_polyline[:-1], dtype=torch.float32)
            target_seq = torch.tensor(padded_polyline[1:], dtype=torch.float32)            

            self.realData.append((input_seq,target_seq,self.data[i][1]))

            if i % 100 == 0:
                print(str(i) + "/" + str(len(self.data)) + " prepared")

        print("Data ready")

    def normalize_data(self):
        """
        Normalizing coordinates x and y in SVG format. Pen-up/Pen-down flag remains untouched.
        Text is not modified either.
        """
        # Extracting all x,y coordinates
        all_points = np.concatenate(
            [np.array(seq[0])[:, :2] for seq in self.data], axis=0
        )
        self.mean = np.mean(all_points, axis=0)
        self.std = np.std(all_points, axis=0)

        # Take Mean And Std Dev to write them down to file
        with open('norm.txt', 'w') as file:
            np.savetxt(file, np.column_stack((self.mean, self.std)))

        # Normalize x, y in `polylines`
        for i in range(len(self.data)):
            polylines, text = self.data[i]
            normalized_polylines = [
                [
                    (point[0] - self.mean[0]) / self.std[0],  # Normalizing x
                    (point[1] - self.mean[1]) / self.std[1],  # Normalizing y
                    point[2],  # Flag remains the same
                ]
                for point in polylines
            ]
            # Updating Normalized Data
            self.data[i] = (normalized_polylines, text)

    def __len__(self):
        """
        Returns number of files in dataset.
        """
        print(len(self.realData))
        return len(self.realData)

    def pad_sequence(self, sequence, max_length):
        """
        Fill sequence with [0, 0, 0] for given size
        """
        sequence_length = len(sequence)
        if sequence_length < max_length:
            padding = [[0, 0, 0]] * (max_length - sequence_length)  # Adding zero timesteps
            sequence.extend(padding)
        return sequence[:max_length]  # Trim to max_length (for safety)

    def pad_alphabet(self, sequence, max_length):
        sequence_length = len(sequence)
        element_size = len(sequence[0]) if sequence else 0
        if sequence_length < max_length:
            padding = [[0] * element_size] * (max_length - sequence_length)  # Adding zero timesteps
            sequence.extend(padding)
        return sequence[:max_length]  # Trim to max_length (for safety)

    def pad_end_probability(self, sequence, max_length):
        sequence_length = len(sequence)
        if sequence_length < max_length:
            padding = [0] * (max_length - sequence_length)  # Adding zero timesteps
            sequence.extend(padding)
        return sequence[:max_length]  # Trim to max_length (for safety)

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
    coords = np.copy(coords)  # Copying base coords

    # Splitting X and Y
    X = coords[:, 0].reshape(-1, 1)
    Y = coords[:, 1].reshape(-1, 1)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack([ones, X])

    # Calculating offset and slope
    XtX = X.T @ X
    XtY = X.T @ Y
    coeffs = np.linalg.solve(XtX, XtY).squeeze()
    offset, slope = coeffs[0], coeffs[1]

    # Calculating angle and rotation matrix
    theta = np.arctan(slope)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotating the cords
    coords[:, :2] = coords[:, :2] @ rotation_matrix - offset

    return coords

def normalize(offsets):
    """
    Normalizes strokes to median unit norm using NumPy.
    """
    offsets = np.copy(offsets)  # Copying data

    # Calculating median norm
    norms = np.linalg.norm(offsets[:, :2], axis=1)
    median_norm = np.median(norms)

    # Normalizing offsets
    offsets[:, :2] /= median_norm

    return offsets

'''
    Convering points to offsets
    And adding [0, 0, 1] as starting point

    We consider whole sequence as if it would start from [0,0] and went on from there.
    It does not matter where exactly you have started.
    The first point will be [0,0] and the following ones will be shifted by the calculated offset.
'''
def coords_to_offsets(coords):
    """
    Convert from coordinates to offsets.
    """
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    offsets = np.concatenate([np.array([[0, 0, 1]]), offsets], axis=0)
    return offsets