import requests
from bs4 import BeautifulSoup

# URL strony
url = "https://pl.wiktionary.org/wiki/Indeks:Polski_-_Najpopularniejsze_s%C5%82owa_1-10000_wersja_Jerzego_Kazojcia"

# Pobranie strony
response = requests.get(url)
response.raise_for_status()  # Wyrzuci błąd, jeśli nie uda się pobrać strony

# Parsowanie strony HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Znalezienie sekcji z danymi
data_section = soup.find('div', {'id': 'mw-content-text'})

# Pobieranie tekstu i wyodrębnianie słów i liczby wystąpień
text = data_section.get_text()
lines = text.splitlines()

# Słownik do przechowywania słów i ich liczby wystąpień
word_counts = {}

# Wyciąganie danych z linii
for line in lines:
    pairs = line.split()
    for pair in pairs:
        if '=' in pair:
            word, _, count = pair.partition('=')  # Rozdziel na pierwszym "="
            try:
                word_counts[word] = int(count)
            except ValueError:
                # Pomijamy przypadki, gdy liczba jest niepoprawna
                continue

# Sortowanie słów według liczby wystąpień
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Zapis słów do pliku
with open('test.txt', 'w', encoding='utf-8') as file:
    for word, _ in sorted_words:
        file.write(f"{word}\n")

print("Słowa zostały zapisane do pliku test.txt!")
