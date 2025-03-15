import os
import re
import csv
import glob

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

def main():
    # Path to the directory containing model files
    models_dir = "last_models"
    
    # Output CSV file
    output_csv = "model_metrics.csv"
    
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
    
    print(f"Successfully extracted data from {len(model_data)} model files.")
    print(f"Data saved to {output_csv}")

main()
