import csv
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils import get_column_letter
import pathlib

def main():
    # Ścieżki do plików
    current_dir = pathlib.Path(__file__).parent.resolve()
    csv_filename = current_dir / "model_metrics.csv"
    excel_filename = current_dir / "model_metrics.xlsx"

    csv_filename = csv_filename.resolve()
    excel_filename = excel_filename.resolve()

    def clean_and_convert(value):
        return float(value.replace('\'', ''))

    # Wczytaj dane z pliku CSV
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # Utwórz nowy plik Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Wyniki"


    for index,row in enumerate(rows):
        if(index == 0):
            # Dodaj nagłówki kolumn
            ws.append(row)
            continue
        # Zamień wartości w kolumnach B, C, D na liczby zmiennoprzecinkowe (usuń przecinki)
        row[1] = clean_and_convert(row[1])  # kolumna B (epoch)
        row[2] = clean_and_convert(row[2])  # kolumna C (train_loss)
        row[3] = clean_and_convert(row[3])  # kolumna D (val_loss)
        ws.append(row)

    # Dodaj nagłówek kolumny E
    ws["E1"] = "avg_train_loss"
    ws["F1"] = "avg_val_loss"

    # Dodaj formuły średniej narastającej
    start_row = 2
    for i in range(start_row, len(rows) + 1):  # i to numer wiersza
        range_str_train = f"C{start_row}:C{i}"
        range_str_val = f"D{start_row}:D{i}"
        
        ws[f"E{i}"] = f"=AVERAGE({range_str_train})"
        ws[f"F{i}"] = f"=AVERAGE({range_str_val})"
        

    # Dodaj wykres (line chart)
    chart = LineChart()
    chart.title = "Wykres Loss"
    data = Reference(ws, min_col=3, min_row=1, max_col=4, max_row=len(rows)+1)
    epoki = Reference(ws, min_col=2, min_row=2, max_col=2, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "H2")  # Dodaj wykres do komórki H2

    chart = LineChart()
    chart.title = "Wykres AVG Loss"
    data = Reference(ws, min_col=5, min_row=1, max_col=6, max_row=len(rows)+1)
    epoki = Reference(ws, min_col=2, min_row=2, max_col=2, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "H17")  # Dodaj wykres do komórki H2


    # Zapisz plik Excel
    wb.save(excel_filename)
    print(f"Zapisano dane z wykresem do {excel_filename}")

    # Zapisz plik Excel
    wb.save(excel_filename)
    print(f"Zapisano dane do {excel_filename}")
    
if __name__ == "__main__":
    main()