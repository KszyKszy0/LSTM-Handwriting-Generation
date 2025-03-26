import os

def rename_files_to_numbers(directory, extension):
    """
    Przechodzi przez wszystkie pliki w katalogu z określonym rozszerzeniem,
    sortuje je alfabetycznie, a następnie zmienia ich nazwy na numery.

    Args:
        directory (str): Ścieżka do katalogu z plikami.
        extension (str): Rozszerzenie plików (np. ".txt").
    """
    # Pobierz listę plików o określonym rozszerzeniu i posortuj alfabetycznie
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    files.sort()

    # Iteracja po posortowanych plikach i zmiana nazw
    for i, filename in enumerate(files, start=1):
        # Ścieżka oryginalnego pliku
        old_path = os.path.join(directory, filename)
        # Nowa nazwa pliku (numer + rozszerzenie)
        new_filename = f"{i}{extension}"
        new_path = os.path.join(directory, new_filename)

        # Zmień nazwę pliku
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")


directory = "output/"
extension = ".svg"

rename_files_to_numbers(directory, extension)