import subprocess
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from gcode.gcodeplot import mainConvert
import time

load_dotenv()

# Pobierz ścieżkę do Inkscape z .env
INKSCAPE_BINARY = os.getenv("INKSCAPE_BINARY")

# Sprawdź, czy zmienna została załadowana
if not INKSCAPE_BINARY:
    raise EnvironmentError("Brak zmiennej INKSCAPE_BINARY w pliku .env. Ustaw ścieżkę do programu Inkscape.")

SVG_PATH = Path(__file__).resolve().parents[1] / "output" / "handwriting.svg"


def force_a6_landscape(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    root.set("width", "148mm")
    root.set("height", "105mm")
    root.set("viewBox", "0 0 740 525")

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def open_in_inkscape_and_slice():
    """
    1. Otwiera handwriting.svg w Inkscape
    2. Ustawia płótno A6 poziome (148x105 mm)
    3. Pozwala użytkownikowi edytować plik
    4. Po zamknięciu Inkscape automatycznie wywołuje slice()
    """
    

    if not SVG_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku SVG: {SVG_PATH}")
    
    force_a6_landscape(SVG_PATH)
    
    time.sleep(1)
    

    inkscape_cmd = [
        INKSCAPE_BINARY,
        str(SVG_PATH),
        "--with-gui"
    ]

    print("Otwieram Inkscape…")
    process = subprocess.Popen(inkscape_cmd)

    process.wait()

    print("Inkscape zamknięty")

    # Upewniamy się, że plik nadal istnieje
    if not SVG_PATH.exists():
        raise RuntimeError("Plik SVG zniknął po zamknięciu Inkscape")

    print("Uruchamiam gcode converter")
    slice()


def slice():
    mainConvert()
    print("Done")
    
