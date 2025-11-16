import os
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np

def getDirs(__file__):
    cwdir = os.path.abspath(os.getcwd())
    cwdir += "/"
    filedir = os.path.abspath(os.path.dirname(__file__))
    filedir += "/"
    return cwdir, filedir

folder_paths = ["../data/output", "../data/mwoutput", "../data/poloutput", "../data/hibru", "../data/augmented"]  # Lista ścieżek do folderów

cwdir, filedir = getDirs(__file__)

# Funkcja do wczytywania danych z wielu folderów
def load_from_folders(folder_paths):
    all_svg_files = []
    all_text_files = []
    
    for folder_path in folder_paths:
        # Ścieżka do pliku z tekstami dla bieżącego folderu
        folder_path = os.path.abspath(filedir + folder_path)
        files_content = f"{folder_path}/files.txt"
        
        # Lista nazw plików z rozszerzeniem .svg z bieżącego folderu
        print("Loading folder: ",files_content)
        svg_files = [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith('.svg')]
        
        # Sortowanie plików SVG numerycznie
        svg_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        all_svg_files.extend(svg_files)
        all_text_files.append(files_content)
    
    return all_svg_files, all_text_files

svg_files, texts = load_from_folders(folder_paths)


chars_list = defaultdict()
for text in texts:
    file = open(text)
    content = file.read()
    for char in content:
        if f"{char}" in chars_list:
            chars_list[f"{char}"] += 1
        else:
            chars_list[f"{char}"] = 1

for char in chars_list:
    if(char == '\n'):
        print("Nowa linia", chars_list[char])
        continue

    if(char == ' '):
        print("Spacja", chars_list[char])
        continue

    print(char, chars_list[char])

plt.bar(chars_list.keys(), chars_list.values())
plt.show()

reciprocal_dict = {k: 1/v for k, v in chars_list.items()}

for char in reciprocal_dict:
    if(char == '\n'):
        print("Nowa linia", reciprocal_dict[char])
        continue

    if(char == ' '):
        print("Spacja", reciprocal_dict[char])
        continue

    print(char, reciprocal_dict[char])

def getImportance(text):
    return math.prod({reciprocal_dict[l] for l in text})
    

importance_words_dict = defaultdict()
for text in texts:
    file = open(text)
    content = file.readlines()
    for line in content:
        importance_words_dict[f"{line}"] = getImportance(line)

for i, word in enumerate(dict(sorted(importance_words_dict.items(), key=lambda item: -item[1]))):
    word_value = getImportance(word) 
    print(word, word_value)

    for textfile in texts:
        file_content = open(textfile)
        text_list = file_content.readlines()
        if word in text_list:
            text_folder = os.path.dirname(textfile)
            index = text_list.index(word)

    if i % 10 == 0:
        input("")


