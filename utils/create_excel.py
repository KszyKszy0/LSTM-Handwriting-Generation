import csv
import os
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.chart import LineChart, Reference
from openpyxl.utils import get_column_letter

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
    
    


    for index,row in enumerate(rows):
        if(index == 0):
            # Dodaj nagłówki kolumn
            ws.append(row)
            continue
        # Zamień wartości w kolumnach B, C, D na liczby zmiennoprzecinkowe (usuń przecinki)
        row[1] = clean_and_convert(row[1])  # kolumna B (epoch)
        row[2] = clean_and_convert(row[2])  # kolumna C (batch_size)
        row[3] = clean_and_convert(row[3])  # kolumna C (train_loss)
        row[4] = clean_and_convert(row[4])  # kolumna D (val_loss)
        ws.append(row)

    # Dodaj nagłówek kolumn
    ws["F1"] = "loss_diff"
    ws["G1"] = "avg_train_loss"
    ws["H1"] = "avg_val_loss"

    # Dodaj formuły średniej narastającej
    start_row = 2
    for i in range(start_row, len(rows) + 1):  # i to numer wiersza
        start_row = max(2, i-10)
        range_str_train = f"D{start_row}:D{i}"
        range_str_val = f"E{start_row}:E{i}"
        
        ws[f"G{i}"] = f"=AVERAGE({range_str_train})"
        ws[f"H{i}"] = f"=AVERAGE({range_str_val})"
        ws[f"F{i}"] = f"=E{i}-D{i}"
        
    # Ustaw wyrównanie do środka dla wszystkich komórek
    
    column_widths = [9, 7, 10, 10, 8, 9, 14 , 12]

    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = column_widths[col[0].column - 1]
        for cell in col:
            i = cell.col_idx
            if int(i) in range(4, 9):
                cell.number_format = "0.0000"
            if(cell.col_idx != 1 or cell.row == 1):
                cell.alignment = Alignment(horizontal='center', vertical='center')
        

    # Dodaj wykres (line chart)
    chart = LineChart()
    chart.title = "Wykres Loss Diff"
    data = Reference(ws, min_col=6, min_row=1, max_col=6, max_row=len(rows)+1)
    epoki = Reference(ws, min_col=2, min_row=2, max_col=2, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "I2")  # Dodaj wykres do komórki H2
    
    # Dodaj wykres (line chart)
    chart = LineChart()
    chart.title = "Wykres Loss"
    data = Reference(ws, min_col=4, min_row=1, max_col=5, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "I18")  # Dodaj wykres do komórki H2

    chart = LineChart()
    chart.title = "Wykres AVG Loss"
    data = Reference(ws, min_col=7, min_row=1, max_col=8, max_row=len(rows)+1)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(epoki)
    ws.add_chart(chart, "I34")  # Dodaj wykres do komórki H2


    # Zapisz plik Excel
    wb.save(excel_filename)
    print(f"XLSX saved to {excel_filename}") if printLogs else None
    
if __name__ == "__main__":
    main()