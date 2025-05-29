import subprocess
import requests
import os
import threading
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

# Załaduj dane z pliku .env

def run_generate_and_upload(model_name: str, epoch: int):
    load_dotenv(dotenv_path=os.path.abspath(os.path.join("..", ".env")))
    username = os.getenv("username")
    password = os.getenv("password")
    text = 'Tymczasowy tekst modelu AI, który służy do przeżucia przez Sztuczną Inteligencję'

    def task():
        try:
            # Ścieżka do pliku wynikowego
            output_path = os.path.abspath(os.path.dirname(__file__) + "/../output/handwriting.svg")
            generate = os.path.abspath(os.path.dirname(__file__) + "/generate.py")

            # 1. Uruchomienie skryptu generate.py z parametrem text
            result = subprocess.run(
                ['python', generate, '--nobrowser', text],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"[Błąd generacji] {result.stderr}")
                return

            # 2. Sprawdzenie, czy plik istnieje
            if not os.path.isfile(output_path):
                print(f"[Błąd] Plik nie istnieje: {output_path}")
                return

            # 3. Wysłanie żądania POST z plikiem
            with open(output_path, 'rb') as f:
                response = requests.post(
                    f'https://ai.d0d0.ovh/api/upload-image',
                    data={
                        'model_name': model_name,
                        'epoch': str(epoch)
                    },
                    files={
                        'image': ('handwriting.svg', f)
                    },
                    auth=HTTPBasicAuth(username, password)
                )

            print(f"[Wysłano obraz] Status: {response.status_code}, Treść: {response.text}")

        except Exception as e:
            print(f"[Wyjątek] {str(e)}")

    # Uruchomienie w osobnym wątku
    thread = threading.Thread(target=task)
    thread.start()