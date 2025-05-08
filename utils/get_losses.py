import os, sys
import time
import re
import csv
import glob
import argparse
import argcomplete
from argcomplete.completers import DirectoriesCompleter

# Dodaj katalog główny projektu do PYTHONPATH
PROJECT_ROOT = os.path.abspath(__file__+"/../../")
sys.path.insert(0, PROJECT_ROOT)
from utils.create_excel import main as create_excel_main

def getargs():
    parser = argparse.ArgumentParser(description='Generate CSV+XLSX from model files')

    parser.add_argument('folder', type=str,
                        help='Folder containing model files').completer = DirectoriesCompleter()

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    userdir,filedir = getDirs(__file__)
    return os.path.abspath(userdir + args.folder)

def getDirs(__file__):
    cwdir = os.path.abspath(os.getcwd())
    cwdir += "/"
    filedir = os.path.abspath(os.path.dirname(__file__))
    filedir += "/"
    return cwdir, filedir

def extract_model_info(filename):
    """
    Extract epoch number, train value, and validation value from a model filename.
    Example filename: "epoch597_train2.3587_val2.3990.pth"
    """
    # Create a regex pattern to match the filename components
    pattern = r"epoch(\d+)_train([-]?\d+\.\d+)_val([-]?\d+\.\d+)\.pth"
    match = re.match(pattern, os.path.basename(filename))
    
    if match:
        epoch = int(match.group(1))
        train_value = float(match.group(2))
        val_value = float(match.group(3))
        return epoch, train_value, val_value
    else:
        return None, None, None

    """
    @param folder: abs path
    """
def main(folder, printLog=True):
    """Main function to process model files and generate a CSV file with metrics.

    Args:
        folder (string(path)): Abs path to the folder containing model files.
    """
    
    # Path to the directory containing model files
    models_dir = folder
    userdir,filedir = getDirs(__file__)
    
    print(f"Processing files in directory: {models_dir}") if printLog else None
    
    # Output CSV file
    output_csv = os.path.abspath(filedir + "../output/model_metrics.csv")
    
    # Get all .pth files in the directory
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    
    # Extract information from each file and store in a list
    model_data = []
    for model_file in model_files:
        epoch, train_value, val_value = extract_model_info(model_file)
        # print(epoch, train_value, val_value)
        if epoch is not None:
            model_data.append({
                'filename': os.path.basename(model_file),
                'epoch': epoch,
                'train_loss': train_value,
                'val_loss': val_value
            })
    
    # Sort the data by epoch number
    model_data.sort(key=lambda x: x['epoch'])
    
    # Write the data to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'epoch', 'train_loss', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in model_data:
            writer.writerow(data)
    if printLog:
        print(f"Successfully extracted data from {len(model_data)} model files.")
        print(f"CSV saved to {output_csv}")        
    
    create_excel_main(printLogs=printLog)
    print(f"Successfully saved CSV and XLSX [{len(model_data)}]") if not printLog else None

if __name__ == "__main__":
    folder = getargs()
    main(folder)
