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

        source_path = os.path.join(input_folder, filename)
        dest_path = os.path.join(output_folder, filename)

        if os.path.exists(source_path):
            try:
                shutil.move(source_path, dest_path)
                # print(f"Moved: {filename}") # Uncomment for verbose output
                moved_count += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
                error_count += 1
        else:
            # print(f"Not found in input folder: {filename}") # Uncomment for verbose output
            not_found_count += 1

    print("\n--- Summary ---")
    print(f"Files successfully moved: {moved_count}")
    print(f"Files not found in input folder: {not_found_count}")
    if error_count > 0:
        print(f"Errors during moving: {error_count}")
    print("-------------")

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    text_file_list = 'data/eval_human/cluster_data_human_631_781.txt'     # CHANGE THIS: Path to your .txt file with filenames
    source_directory = 'data/eval_human/all_clusters'    # CHANGE THIS: Folder containing the files to move
    destination_directory = 'data/eval_human/human_clusters' # CHANGE THIS: Folder to move files into
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
