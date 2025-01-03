import unidecode

# Pliki wejściowe i wyjściowe
INPUT_FILE = "../słowniki/test.txt"  # Oryginalny plik słownika
OUTPUT_FILE = "../words.txt"   # Plik docelowy z wyrazami bepipz polskich znaków

def remove_polish_characters(word):
    """
    Usuwa polskie znaki z wyrazu.
    """
    return unidecode.unidecode(word)

def filter_words(input_file, output_file):
    """
    Przepisuje słowa bez polskich znaków do nowego pliku.
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            word = line.strip()
            # Usuwamy znaki diakrytyczne i sprawdzamy, czy słowo jest identyczne po tej operacji
            if word == remove_polish_characters(word):
                outfile.write(word + "\n")

# Wykonanie skryptu
try:
    filter_words(INPUT_FILE, OUTPUT_FILE)
    print(f"Przetworzono słownik. Wyrazy bez polskich znaków zapisano w pliku {OUTPUT_FILE}.")
except FileNotFoundError:
    print(f"Plik {INPUT_FILE} nie istnieje! Upewnij się, że podany plik jest w katalogu.")
except Exception as e:
    print(f"Wystąpił błąd: {e}")
