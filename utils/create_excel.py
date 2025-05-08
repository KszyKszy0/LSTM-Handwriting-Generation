import csv
import os
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.chart import LineChart, Reference

def getDirs(__file__):
    cwdir = os.path.abspath(os.getcwd() + "/../output/")
    filedir = os.path.abspath(os.path.dirname(__file__) + "/../output/")
    return cwdir, filedir
    
def main(printLogs=True):
    
    # Ścieżki do plików
    userdir,filedir = getDirs(__file__)

    csv_filename = os.path.join(filedir, 'model_metrics.csv')
    excel_filename = os.path.join(filedir, 'model_metrics.xlsx')

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
    
    kolumny = ['C', 'D', 'E', 'F', 'G']
    for kolumna in kolumny:
        ws.column_dimensions[kolumna].number_format = "0.00000"


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

    # Dodaj nagłówek kolumn
    ws["E1"] = "loss_diff"
    ws["F1"] = "avg_train_loss"
    ws["G1"] = "avg_val_loss"

    # Dodaj formuły średniej narastającej
    start_row = 2
    for i in range(start_row, len(rows) + 1):  # i to numer wiersza
        start_row = max(2, i-10)
        range_str_train = f"C{start_row}:C{i}"
        range_str_val = f"D{start_row}:D{i}"
        
        ws[f"F{i}"] = f"=AVERAGE({range_str_train})"
        ws[f"G{i}"] = f"=AVERAGE({range_str_val})"
        ws[f"E{i}"] = f"=D{i}-C{i}"
        

    # Dodaj wykres (line chart)
    chart = LineChart()
    chart.title = "Wykres Loss Diff"
    data = Reference(ws, min_col=5, min_row=1, max_col=5, max_row=len(rows)+1)
    epoki = Reference(ws, min_col=2, min_row=2, max_col=2, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "H2")  # Dodaj wykres do komórki H2
    
    # Dodaj wykres (line chart)
    chart = LineChart()
    chart.title = "Wykres Loss"
    data = Reference(ws, min_col=3, min_row=1, max_col=4, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "H18")  # Dodaj wykres do komórki H2

    chart = LineChart()
    chart.title = "Wykres AVG Loss"
    data = Reference(ws, min_col=6, min_row=1, max_col=7, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "H34")  # Dodaj wykres do komórki H2


    # Zapisz plik Excel
    wb.save(excel_filename)
    print(f"XLSX saved to {excel_filename}") if printLogs else None
    
if __name__ == "__main__":
    main()