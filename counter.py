from collections import Counter

def count_top_words_in_file(filename, top_n=10):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            words = [line.strip() for line in file if line.strip()]
        word_counts = Counter(words)
        
        # Posortowanie wszystkich wyrazów
        sorted_words = word_counts.most_common()
        
        # Pobranie top_n najczęściej występujących wyrazów
        top_words = sorted_words[:top_n]
        
        # Sprawdzenie ostatniego elementu w top_n i znalezienie pozostałych wyrazów o tej samej liczbie
        last_count = top_words[-1][1] if top_words else None
        remaining_words = [word for word in sorted_words[top_n:] if word[1] == last_count]
        
        print(f"Top {top_n} najczęściej występujących wyrazów:")
        for word, count in top_words:
            print(f"{word}: {count}")
        
        # Wyświetlenie pozostałych wyrazów o tej samej liczbie wystąpień
        if remaining_words:
            for word, count in remaining_words:
                print(f"{word}: {count}")
        
    except FileNotFoundError:
        print(f"Plik {filename} nie został znaleziony.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    count_top_words_in_file("output/files.txt")
