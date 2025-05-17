import os
import re

def getDirs(__file__):
    cwdir = os.path.abspath(os.getcwd()) + "/"
    filedir = os.path.abspath(os.path.dirname(__file__)) + "/"
    return cwdir, filedir

def main():
    cwdir, filedir = getDirs(__file__)

    # Ścieżka do folderu z modelami
    models_dir = os.path.abspath(os.path.join(filedir, "../models/"))

    # Pobieramy listę plików z katalogu
    files = os.listdir(models_dir)

    # Wyrażenie regularne dopasowujące pliki o wzorze epoch{liczba}_coś
    regex = r"epoch(\d+)_.*"

    # Filtrujemy tylko pliki zgodne z wzorcem i parsujemy numer epoki
    filtered_files = [(int(re.match(regex, f).group(1)), f) for f in files if re.match(regex, f)]

    highest_epoch_file = max(filtered_files, key=lambda x: x[0])[0]

    # Usuwamy pliki, których numer epoki jest parzysty
    for epoch_num, filename in filtered_files:
        if epoch_num % 2 != 0 and epoch_num > 30 and epoch_num < highest_epoch_file-30:
            file_path = os.path.join(models_dir, filename)
            try:
                os.remove(file_path)
                print(f"Usunięto plik: {filename}")
            except Exception as e:
                print(f"Nie udało się usunąć {filename}: {e}")

if(__name__ == "__main__"):
    main()