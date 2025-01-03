import os
import random
import tkinter as tk
from tkinter import messagebox
import svgwrite
import re


# usunąć alert
# susnąć loga

# Ustawienia globalne
OUTPUT_DIR = "output"  # Katalog na pliki SVG
LOG_FILE = "output/files.txt"  # Plik z listą nazw plików
WORDS_FILE = "words.txt"  # Plik ze słowami do losowania
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 150

def get_next_file_number(directory, prefix, extension):
    """
    Znajduje największy numer pliku z określonym prefixem i rozszerzeniem w katalogu,
    a następnie zwraca następny numer, który można użyć do nazwania nowego pliku.

    Args:
        directory (str): Ścieżka do katalogu, w którym znajdują się pliki.
        prefix (str): Prefiks plików (np. "plik_").
        extension (str): Rozszerzenie plików (np. ".txt").

    Returns:
        int: Następny dostępny numer pliku.
    """
    # Regularne wyrażenie do wyszukiwania numerów w nazwach plików
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(extension)}$")
    max_number = 0

    # Przejrzyj pliki w katalogu
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number

    # Zwróć następny numer
    return max_number + 1

def load_words():
    if not os.path.exists(WORDS_FILE):
        raise FileNotFoundError(f"Plik \"{WORDS_FILE}\" nie istnieje!")
    with open(WORDS_FILE, "r") as f:
        return [word.strip() for word in f.readlines() if word.strip()]

def save_svg(canvas, filename):
    print(f"Zapisywanie SVG do pliku: {filename}")
    dwg = svgwrite.Drawing(filename, size=(CANVAS_WIDTH, CANVAS_HEIGHT))
    for line in canvas.lines:
        if len(line) > 1:
            dwg.add(dwg.polyline(points=line, stroke="black", fill="none", stroke_width=2))
    dwg.save()

# Klasa aplikacji
class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting AI Tool")

        # Zmienna do rysowania
        self.lines = []
        self.current_line = []

        # Słowa
        self.words = load_words()
        print(f"Załadowano słowa: {self.words}")
        self.current_word = ""

        # Numeracja plików
        self.file_number = get_next_file_number(OUTPUT_DIR,"",".svg")
        print(f"Następny numer pliku: {self.file_number}")

        # Interfejs użytkownika
        self.label = tk.Label(root, text="", font=("Arial", 24))
        self.label.pack(pady=10)
        self.update_word()

        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white", cursor="tcross")
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_line)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_canvas)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(self.button_frame, text="Zapisz", command=self.save_canvas)
        self.save_button.pack(side=tk.LEFT, padx=10)

        # Obsługa skrótów klawiszowych
        self.root.bind("<r>", lambda event: self.reset_canvas())
        self.root.bind("<Return>", lambda event: self.next_word())

    def update_word(self):
        self.current_word = random.choice(self.words)
        print(f"Wylosowane słowo: {self.current_word}")
        self.label.config(text=f"Przepisz: {self.current_word}")
        
    def draw(self, event):
        x, y = event.x, event.y
        self.current_line.append((x, y))
        if len(self.current_line) > 1:
            self.canvas.create_line(self.current_line[-2], self.current_line[-1], fill="black", width=2)

    def end_line(self, event):
        if self.current_line:
            # print(f"Zakończono linię: {self.current_line}")
            self.lines.append(self.current_line)
            self.current_line = []

    def reset_canvas(self):
        print("Resetowanie pola rysowania.")
        self.canvas.delete("all")
        self.lines = []
        self.current_line = []

    def save_canvas(self):
        if not self.lines:
            print("Brak danych do zapisania!")
            messagebox.showwarning("Brak danych", "Pole rysowania jest puste!")
            return


        filename = f"{self.file_number}.svg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Sprawdzanie czy plik istnieje i dodawanie numeracji
        # counter = 1
        # while os.path.exists(filepath):
        #     filename = f"{self.file_number} {counter}.svg"
        #     filepath = os.path.join(OUTPUT_DIR, filename)
        #     counter += 1

        # Zapis pliku
        save_svg(self, filepath)

        # Logowanie
        with open(LOG_FILE, "a") as log:
            log.write(self.current_word + "\n")

        print(f"Zapisano plik: {filename}, słowo: {self.current_word}")
        # messagebox.showinfo("Zapisano", f"Plik zapisany jako {filename}")

        # Aktualizacja stanu aplikacji
        self.file_number += 1
        self.reset_canvas()
        self.update_word()

    def next_word(self):
        self.reset_canvas()
        self.update_word()

# Uruchomienie aplikacji
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
