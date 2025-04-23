import pandas as pd
import os
import shutil

def move_files_from_excel(excel_path, input_folder, output_folder, frame_col='frame', cluster_col='cluster', sheet_name=0):
    """
    Reads frame and cluster numbers from a specific Excel sheet, constructs filenames,
    and moves corresponding .bin files from an input folder to an output folder.

    Args:
        excel_path (str): Path to the Excel file (.xlsx or .xls).
        input_folder (str): Path to the folder containing the source .bin files.
        output_folder (str): Path to the folder where files should be moved.
        frame_col (str): Name of the column containing frame numbers in the Excel sheet.
        cluster_col (str): Name of the column containing cluster numbers in the Excel sheet.
        sheet_name (str or int, optional): Name or zero-based index of the sheet to read.
                                           Defaults to 0 (the first sheet).
    """
    try:
        # Read the specified sheet
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    moved_count = 0
    not_found_count = 0

    print(f"Processing file: {excel_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    for index, row in df.iterrows():
        try:
            frame_num = int(row[frame_col])
            cluster_num = int(row[cluster_col])

            # Format the filename (adjust padding/format if needed)
            # Assumes 6-digit zero-padded frame number, e.g., 000311
            filename = f"cluster_frame_{frame_num:06d}_cluster_{cluster_num}.bin"
            source_path = os.path.join(input_folder, filename)
            dest_path = os.path.join(output_folder, filename)

            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, dest_path)
                    print(f"Moved: {filename}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            else:
                print(f"Not found in input folder: {filename}")
                not_found_count += 1

        except KeyError as e:
            print(f"Error: Column '{e}' not found in Excel file. Please check 'frame_col' and 'cluster_col' arguments.")
            return
        except ValueError:
            print(f"Warning: Skipping row {index + 2} due to non-integer value in frame or cluster column.")
            continue
        except Exception as e:
            print(f"Error processing row {index + 2}: {e}")
            continue

    print("\n--- Summary ---")
    print(f"Files successfully moved: {moved_count}")
    print(f"Files not found in input folder: {not_found_count}")
    print("-------------")


if __name__ == "__main__":
    # --- Configuration ---
    excel_file_path = 'data/eval_human/cluster_labels.xlsx' # CHANGE THIS
    source_directory = 'data/eval_human/all_clusters' # CHANGE THIS
    destination_directory = 'data/eval_human/human_clusters' # CHANGE THIS

    frame_column_name = 'Frame'
    cluster_column_name = 'Cluster'
    sheet_to_read = 1
    # -------------------

    # Make sure the paths exist before running
    if not os.path.exists(excel_file_path):
         print(f"Error: Please update 'excel_file_path' to a valid path.")
    elif not os.path.exists(source_directory):
         print(f"Error: Please update 'source_directory' to a valid path.")
    else:
        move_files_from_excel(
            excel_file_path,
            source_directory,
            destination_directory,
            frame_col=frame_column_name,
            cluster_col=cluster_column_name,
            sheet_name=sheet_to_read
        )
