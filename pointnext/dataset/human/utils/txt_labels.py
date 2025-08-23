import os
import shutil

def move_files_from_txt(txt_path, input_folder, output_folder):
    """
    Reads filenames from a text file (one filename per line) and moves
    corresponding files from an input folder to an output folder.

    Args:
        txt_path (str): Path to the text file containing filenames.
        input_folder (str): Path to the folder containing the source files.
        output_folder (str): Path to the folder where files should be moved.
    """
    try:
        with open(txt_path, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()] # Read and strip whitespace/empty lines
    except FileNotFoundError:
        print(f"Error: Text file not found at {txt_path}")
        return
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        return

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    moved_count = 0
    not_found_count = 0
    error_count = 0

    print(f"Processing file list: {txt_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    for filename in filenames:
        if not filename: # Skip empty lines potentially missed by initial strip
            continue

        # Handle both relative filenames and absolute paths
        # If filename contains path separators, extract just the filename
        if os.sep in filename or '/' in filename:
            actual_filename = os.path.basename(filename)
            print(f"Extracted filename from path: {actual_filename}")
        else:
            actual_filename = filename

        source_path = os.path.join(input_folder, actual_filename)
        dest_path = os.path.join(output_folder, actual_filename)

        if os.path.exists(source_path):
            try:
                shutil.move(source_path, dest_path)
                print(f"Moved: {actual_filename}") # Uncomment for verbose output
                moved_count += 1
            except Exception as e:
                print(f"Error moving {actual_filename}: {e}")
                error_count += 1
        else:
            print(f"Not found in input folder: {actual_filename}") # Uncomment for verbose output
            not_found_count += 1

    print("\n--- Summary ---")
    print(f"Files successfully moved: {moved_count}")
    print(f"Files not found in input folder: {not_found_count}")
    if error_count > 0:
        print(f"Errors during moving: {error_count}")
    print("-------------")
    
    # Debug: Check destination folder contents
    print(f"\n--- Debug Info ---")
    print(f"Destination folder: {output_folder}")
    if os.path.exists(output_folder):
        dest_files = os.listdir(output_folder)
        print(f"Files in destination folder: {len(dest_files)}")
        if len(dest_files) > 0:
            print(f"First 5 files in destination: {dest_files[:5]}")
    else:
        print("Destination folder does not exist!")
    
    # Check source folder contents
    print(f"\nSource folder: {input_folder}")
    if os.path.exists(input_folder):
        source_files = os.listdir(input_folder)
        print(f"Files remaining in source folder: {len(source_files)}")
        if len(source_files) > 0:
            print(f"First 5 files remaining: {source_files[:5]}")
    else:
        print("Source folder does not exist!")
    print("----------------")

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    text_file_list = 'D:/CMU_Research/Data/labeled_data/labelingforeval_human/ood_clusters.txt'     # CHANGE THIS: Path to your .txt file with filenames
    source_directory = "D:/CMU_Research/Data/labeled_data/human_dataset1/human_clusters"    # CHANGE THIS: Folder containing the files to move
    destination_directory = 'D:/CMU_Research/Data/labeled_data/human_dataset1/ood_clusters' # CHANGE THIS: Folder to move files into
    # -------------------

    # Basic validation before running
    if not os.path.exists(text_file_list):
         print(f"Error: Please update 'text_file_list' to a valid path.")
    elif not os.path.exists(source_directory):
         print(f"Error: Please update 'source_directory' to a valid path.")
    else:
        move_files_from_txt(
            text_file_list,
            source_directory,
            destination_directory
        )

    print("Script finished.")
