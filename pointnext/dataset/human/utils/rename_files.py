import os
import sys
from pathlib import Path

def rename_files_to_human_cluster(folder_path, file_extension=None):
    """
    Rename all files in a folder to follow the pattern human_cluster_1, human_cluster_2, etc.
    
    Args:
        folder_path (str): Path to the folder containing files to rename
        file_extension (str, optional): Only rename files with this extension (e.g., '.bin', '.txt')
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    # Get all files in the directory
    if file_extension:
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == file_extension.lower()]
    else:
        files = [f for f in folder.iterdir() if f.is_file()]
    
    if not files:
        print(f"No files found in '{folder_path}'")
        if file_extension:
            print(f"with extension '{file_extension}'")
        return
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    print(f"Found {len(files)} files to rename:")
    for i, file in enumerate(files, 1):
        print(f"  {file.name}")
    
    # Ask for confirmation
    response = input(f"\nDo you want to rename these files to human_cluster_1, human_cluster_2, etc.? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Rename files
    renamed_count = 0
    for i, file in enumerate(files, 1):
        new_name = f"human_cluster_{i}{file.suffix}"
        new_path = folder / new_name
        
        try:
            # Check if target name already exists
            if new_path.exists():
                print(f"Warning: '{new_name}' already exists, skipping '{file.name}'")
                continue
            
            file.rename(new_path)
            print(f"Renamed: {file.name} -> {new_name}")
            renamed_count += 1
            
        except Exception as e:
            print(f"Error renaming '{file.name}': {e}")
    
    print(f"\nSuccessfully renamed {renamed_count} out of {len(files)} files.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python rename_files.py <folder_path> [file_extension]")
        print("Example: python rename_files.py 'd:/CMU_Research/Data/labeled_data/false_clusters' .bin")
        return
    
    folder_path = sys.argv[1]
    file_extension = sys.argv[2] if len(sys.argv) > 2 else None
    
    rename_files_to_human_cluster(folder_path, file_extension)

if __name__ == "__main__":
    main()
