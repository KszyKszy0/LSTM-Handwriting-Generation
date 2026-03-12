import os
from collections import Counter, defaultdict

def load_text_content_from_files_txt(files_txt_path, svg_files_in_folder):
    """
    Load text content from files.txt where each line corresponds to each SVG file.
    
    Args:
        files_txt_path: Path to files.txt
        svg_files_in_folder: List of SVG files from this folder (sorted)
    
    Returns:
        Dictionary mapping SVG file path to its text content
    """
    file_content_map = {}
    
    if not os.path.exists(files_txt_path):
        print(f"Warning: {files_txt_path} not found")
        return file_content_map
    
    try:
        with open(files_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Match each line with corresponding SVG file
        for i, svg_file in enumerate(svg_files_in_folder):
            if i < len(lines):
                file_content_map[svg_file] = lines[i].strip()
            else:
                print(f"Warning: No text content for {svg_file}")
        
        return file_content_map
    
    except Exception as e:
        print(f"Error reading {files_txt_path}: {e}")
        return file_content_map

def calculate_character_counts(text):
    """Count all characters in the text."""
    return Counter(text)

def select_balanced_files(all_svg_files, folder_paths, filedir='', target_count=300):
    """
    Select files to balance character distribution.
    
    Args:
        all_svg_files: List of all SVG file paths (sorted)
        folder_paths: List of folder paths
        filedir: Base directory
        target_count: Target count for each character type
    
    Returns:
        List of selected file paths
    """
    
    # Build mapping: SVG file -> character counts
    file_char_counts = {}
    
    # Process each folder
    for folder_path in folder_paths:
        folder_path_abs = os.path.abspath(filedir + folder_path)
        files_txt_path = f"{folder_path_abs}/files.txt"
        
        # Get SVG files from this folder
        svg_files_in_folder = [f for f in all_svg_files 
                               if os.path.dirname(f) == folder_path_abs]
        
        # Load text content
        content_map = load_text_content_from_files_txt(files_txt_path, svg_files_in_folder)
        
        # Calculate character counts for each file
        for svg_file, text_content in content_map.items():
            char_counts = calculate_character_counts(text_content)
            file_char_counts[svg_file] = char_counts
    
    print(f"Loaded character counts for {len(file_char_counts)} files\n")
    
    if not file_char_counts:
        print("Error: No character counts found!")
        return []
    
    # Initialize tracking
    current_counts = Counter()
    selected_files = []
    remaining_files = list(file_char_counts.keys())
    
    def calculate_score(file_path):
        """Calculate how well a file balances the current distribution."""
        file_counts = file_char_counts[file_path]
        score = 0
        
        for char, count in file_counts.items():
            current = current_counts.get(char, 0)
            deficit = target_count - current
            
            # Reward characters we need, penalize characters we have enough of
            if deficit > 0:
                score += min(count, deficit)
            else:
                score -= count * 0.5  # Penalty for oversupply
        
        return score
    
    # Greedy selection: pick files that best balance the distribution
    print("Starting file selection...\n")
    
    while remaining_files:
        # Score all remaining files
        scored_files = [(f, calculate_score(f)) for f in remaining_files]
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        best_file, best_score = scored_files[0]
        
        # Stop if adding more files would oversaturate
        if best_score <= 0 and len(selected_files) > 0:
            # Check if we're close to target
            if current_counts:
                avg_count = sum(current_counts.values()) / len(current_counts)
                if avg_count >= target_count * 0.8:  # 80% of target is acceptable
                    print(f"Stopping: reached 80% of target (avg: {avg_count:.1f})")
                    break
        
        # Add the best file
        selected_files.append(best_file)
        current_counts.update(file_char_counts[best_file])
        remaining_files.remove(best_file)
        
        # Print progress every 10 files
        if len(selected_files) % 10 == 0:
            unique_chars = len(current_counts)
            avg_count = sum(current_counts.values()) / max(unique_chars, 1)
            print(f"Selected {len(selected_files)} files | "
                  f"{unique_chars} unique chars | "
                  f"avg count: {avg_count:.1f}")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"FINAL SELECTION STATISTICS")
    print(f"{'='*60}")
    print(f"Selected files: {len(selected_files)} out of {len(file_char_counts)}")
    print(f"Unique characters: {len(current_counts)}")
    
    # Show distribution for common character types
    char_types = {
        'lowercase': sum(current_counts.get(c, 0) for c in 'abcdefghijklmnopqrstuvwxyz'),
        'uppercase': sum(current_counts.get(c, 0) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'digits': sum(current_counts.get(c, 0) for c in '0123456789'),
        'punctuation': sum(current_counts.get(c, 0) for c in '.,!?;:\'"'),
        'special': sum(current_counts.get(c, 0) for c in '<>[]{}()/-+=*&^%$#@'),
        'space': current_counts.get(' ', 0),
    }
    
    print("\nCharacter type distribution:")
    for char_type, count in char_types.items():
        print(f"  {char_type:15s}: {count:6d}")
    
    # Show individual character counts (top 30)
    print("\nTop 30 individual character counts:")
    for i, (char, count) in enumerate(current_counts.most_common(30), 1):
        if char == ' ':
            char_repr = '[space]'
        elif char == '\n':
            char_repr = '[newline]'
        elif char == '\t':
            char_repr = '[tab]'
        else:
            char_repr = char
        print(f"  {i:2d}. '{char_repr}': {count}")
    
    # Check how close we are to target
    print(f"\nCharacters within 80%-120% of target ({target_count}):")
    in_range = sum(1 for count in current_counts.values() 
                   if target_count * 0.8 <= count <= target_count * 1.2)
    print(f"  {in_range} out of {len(current_counts)} characters")
    
    return selected_files

def load_from_folders(folder_paths, filedir=''):
    """Load SVG files and files.txt paths from multiple folders."""
    all_svg_files = []
    all_text_files = []
    
    for folder_path in folder_paths:
        folder_path_abs = os.path.abspath(filedir + folder_path)
        files_content = f"{folder_path_abs}/files.txt"
        
        print(f"Loading folder: {folder_path_abs}")
        
        if not os.path.exists(folder_path_abs):
            print(f"  Warning: Folder does not exist, skipping")
            continue
        
        svg_files = [f"{folder_path_abs}/{file}" for file in os.listdir(folder_path_abs) 
                     if file.endswith('.svg')]
        
        # Sort numerically
        svg_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        print(f"  Found {len(svg_files)} SVG files")
        
        all_svg_files.extend(svg_files)
        all_text_files.append(files_content)
    
    print(f"\nTotal SVG files: {len(all_svg_files)}\n")
    return all_svg_files, all_text_files

# Example usage:
if __name__ == "__main__":
    folder_paths = ["data/output", "data/mwoutput", "data/poloutput", 
                    "data/hibru", "data/augmented"]
    
    # Load all files
    svg_files, text_files = load_from_folders(folder_paths)
    
    # Select balanced subset
    selected_files = select_balanced_files(svg_files, folder_paths, target_count=700)
    
    with open("core/filter.txt", 'a') as file:
        print(f"\n{'='*60}")
        print("SELECTED FILES:")
        print(f"{'='*60}")
        for i, file_path in enumerate(selected_files, 1):
            # print(f"{i:4d}. {file_path}")
            file.write(file_path + "\n")