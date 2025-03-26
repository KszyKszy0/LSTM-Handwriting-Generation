def sort_file_lines(input_file, output_file=None):
    """
    Sortuje linie w pliku alfabetycznie.

    Args:
        input_file (str): Ścieżka do pliku wejściowego.
        output_file (str): Ścieżka do pliku wyjściowego. Jeśli None, nadpisuje plik wejściowy.
    """
    # Odczytaj linie z pliku
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Posortuj linie alfabetycznie
    sorted_lines = sorted(lines)

    # Zapisz wynik do pliku (oryginalnego lub nowego)
    if output_file is None:
        output_file = input_file

    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(sorted_lines)

    print(f"Plik został posortowany i zapisany w: {output_file}")

# Przykład użycia
input_file = "output/files.txt"
output_file = None  # Podaj inną nazwę pliku, jeśli chcesz zapisać wynik w nowym pliku

sort_file_lines(input_file, output_file)