import numpy as np
import open3d as o3d
import os
import argparse
import re
from collections import defaultdict
import pandas as pd

# --- Color Definitions ---
# Using RGB tuples [0.0 - 1.0]
TP_HUMAN_COLOR = [0, 1, 0]    # Green: Correctly predicted Human (True=0, Pred=0)
TP_FALSE_COLOR = [0, 0, 1]    # Blue: Correctly predicted False Positive (True=1, Pred=1)
FN_HUMAN_COLOR = [1, 0, 0]    # Red: Human misclassified as False Positive (True=0, Pred=1) - False Negative for Human class
FP_HUMAN_COLOR = [1, 1, 0]    # Yellow: False Positive misclassified as Human (True=1, Pred=0) - False Positive for Human class
UNKNOWN_PRED_COLOR = [0.5, 0.5, 0.5] # Gray: Predicted as Unknown (Pred=-1)
DEFAULT_COLOR = [1, 1, 1]     # White: For cases where prediction is missing (should not happen ideally)

HUMAN_LABEL = 0
FALSE_LABEL = 1
UNKNOWN_LABEL = -1

def load_bin(file_path):
    """Loads points from a .bin file (x, y, z, intensity)."""
    try:
        points = np.fromfile(file_path, dtype=np.float32)
        points = points.reshape(-1, 4)
        return points
    except Exception as e:
        print(f"Error loading bin file {file_path}: {e}")
        return None

def extract_frame_number(filename):
    """Extracts frame number from filename like cluster_frame_XXX_cluster_YYY.bin"""
    match = re.search(r'cluster_frame_(\d+)_cluster_\d+\.bin', filename)
    if match:
        return int(match.group(1))
    print(f"Warning: Could not extract frame number from {filename}")
    return -1 # Return -1 to indicate failure

def extract_cluster_id(filename):
    """Extracts cluster ID from filename like cluster_frame_XXX_cluster_YYY.bin"""
    match = re.search(r'cluster_frame_\d+_cluster_(\d+)\.bin', filename)
    if match:
        return int(match.group(1))
    print(f"Warning: Could not extract cluster ID from {filename}")
    return -1 # Return -1 to indicate failure

def group_by_frame(files):
    """Groups file paths by their extracted frame number."""
    frames = defaultdict(list)
    for file_path in files:
        filename = os.path.basename(file_path)
        frame_num = extract_frame_number(filename)
        if frame_num != -1: # Only add if frame number was extracted successfully
            frames[frame_num].append(file_path)
    
    # Sort frames by frame number
    sorted_frames = sorted(frames.items())
    return sorted_frames

def load_split_file(split_file_path):
    """Loads an ordered list of filenames from a split file."""
    try:
        with open(split_file_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines
            filenames = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(filenames)} filenames from {split_file_path}")
        return filenames
    except FileNotFoundError:
        print(f"Error: Split file not found at {split_file_path}")
        return None
    except Exception as e:
        print(f"Error reading split file {split_file_path}: {e}")
        return None

def load_predictions(predictions_csv_path):
    """Loads predictions from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(predictions_csv_path)
        if 'true_label' not in df.columns or 'predicted_label' not in df.columns:
            print(f"Error: Predictions CSV must contain 'true_label' and 'predicted_label' columns.")
            return None
        print(f"Loaded {len(df)} predictions from {predictions_csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Predictions CSV not found at {predictions_csv_path}")
        return None
    except Exception as e:
        print(f"Error reading predictions CSV {predictions_csv_path}: {e}")
        return None

def map_predictions_to_files(split_filenames, predictions_df):
    """Maps predictions to filenames based on order in the split file."""
    if len(split_filenames) != len(predictions_df):
        print(f"Error: Mismatch between number of files in split ({len(split_filenames)}) and predictions ({len(predictions_df)}).")
        print("Ensure the predictions CSV corresponds exactly to the split file used for evaluation.")
        return None

    mapping = {}
    for i, filename in enumerate(split_filenames):
        basename = os.path.basename(filename) # Use basename as key
        mapping[basename] = {
            'true_label': predictions_df.iloc[i]['true_label'],
            'predicted_label': predictions_df.iloc[i]['predicted_label']
        }
    print(f"Successfully mapped predictions to {len(mapping)} files.")
    return mapping

def get_color_and_label(true_label, predicted_label):
    """Determines the color and descriptive label based on true and predicted labels."""
    if predicted_label == UNKNOWN_LABEL:
        return UNKNOWN_PRED_COLOR, f"Predicted Unknown (True: {true_label})"
    elif true_label == HUMAN_LABEL and predicted_label == HUMAN_LABEL:
        return TP_HUMAN_COLOR, "TP Human"
    elif true_label == FALSE_LABEL and predicted_label == FALSE_LABEL:
        return TP_FALSE_COLOR, "TP False Positive"
    elif true_label == HUMAN_LABEL and predicted_label == FALSE_LABEL:
        return FN_HUMAN_COLOR, "FN Human (Predicted False)"
    elif true_label == FALSE_LABEL and predicted_label == HUMAN_LABEL:
        return FP_HUMAN_COLOR, "FP Human (Predicted Human)"
    else:
        # Should not happen with labels 0, 1, -1 but good to have a fallback
        return DEFAULT_COLOR, f"Other (True: {true_label}, Pred: {predicted_label})"


def visualize_predictions(frames, prediction_mapping, split_name=None):
    """Visualizes point cloud frames, coloring clusters by prediction results."""
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_title = "Prediction Visualization"
    if split_name:
        window_title += f" - {split_name} split"
    vis.create_window(window_name=window_title, width=1280, height=720)
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # State variables
    current_frame_idx = [0]
    current_cluster_idx = [0] # Index within the current frame's clusters
    view_mode = ["frame"] # "frame" or "cluster"
    all_pcds_in_frame = [[]] # Store Pcds of the current frame for cluster view


    # --- Print Legend ---
    print("\n--- Color Legend ---")
    print(f"  Green  ({TP_HUMAN_COLOR}): TP Human (True=0, Pred=0)")
    print(f"  Blue   ({TP_FALSE_COLOR}): TP False Positive (True=1, Pred=1)")
    print(f"  Red    ({FN_HUMAN_COLOR}): FN Human (True=0, Pred=1)")
    print(f"  Yellow ({FP_HUMAN_COLOR}): FP Human (True=1, Pred=0)")
    print(f"  Gray   ({UNKNOWN_PRED_COLOR}): Predicted Unknown (Pred=-1)")
    print(f"  White  ({DEFAULT_COLOR}): Prediction Missing/Other")
    print("--------------------\n")


    def update_visualization(frame_idx):
        """Clears and updates the visualizer with the selected frame/cluster."""
        nonlocal all_pcds_in_frame
        if not (0 <= frame_idx < len(frames)):
            print("Invalid frame index.")
            return False

        vis.clear_geometries()
        vis.add_geometry(coordinate_frame) # Re-add coordinate frame

        frame_num, cluster_files = frames[frame_idx]
        
        cluster_info_list = []
        rendered_pcds = [] # Store pcds added to the visualizer for cluster view logic
        all_pcds_in_frame[0] = [] # Reset pcds for the current frame

        total_points = 0
        
        # Load, color, and potentially add geometry for each cluster in the frame
        for file_path in cluster_files:
            filename = os.path.basename(file_path)
            cluster_id = extract_cluster_id(filename)
            
            points = load_bin(file_path)
            if points is None or len(points) == 0:
                print(f"  Skipping empty or unloadable cluster: {filename}")
                continue

            num_points = len(points)
            total_points += num_points

            # Get prediction results
            pred_data = prediction_mapping.get(filename)
            if pred_data:
                true_label = int(pred_data['true_label'])
                predicted_label = int(pred_data['predicted_label'])
                color, label_str = get_color_and_label(true_label, predicted_label)
            else:
                print(f"  Warning: No prediction found for {filename}. Using default color.")
                color, label_str = DEFAULT_COLOR, "Prediction Missing"
                true_label, predicted_label = "?", "?"


            # Create PointCloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.paint_uniform_color(color) # Color the whole cluster

            # Store cluster info and the pcd object itself
            cluster_info = {
                'id': cluster_id,
                'path': file_path,
                'num_points': num_points,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'label_str': label_str,
                'pcd': pcd
            }
            cluster_info_list.append(cluster_info)
            all_pcds_in_frame[0].append(cluster_info) # Store for cluster view

        # Sort clusters by ID for consistent display order
        cluster_info_list.sort(key=lambda x: x['id'])
        all_pcds_in_frame[0].sort(key=lambda x: x['id'])


        # --- Display Logic (Frame vs Cluster) ---
        if view_mode[0] == "frame":
            print(f"\n--- Frame: {frame_num} (Full View) ---")
            print(f"  Clusters: {len(cluster_info_list)}")
            print(f"  Total points: {total_points}")
            print("  Cluster Details (ID: Points, True->Pred, Status):")
            for info in cluster_info_list:
                vis.add_geometry(info['pcd']) # Add all pcds in frame view
                rendered_pcds.append(info)
                print(f"    Cluster {info['id']}: {info['num_points']} pts, {info['true_label']}->{info['predicted_label']}, {info['label_str']}")

        elif view_mode[0] == "cluster":
            if all_pcds_in_frame[0]: # If there are clusters in this frame
                # Ensure cluster index is valid
                num_clusters_in_frame = len(all_pcds_in_frame[0])
                if current_cluster_idx[0] >= num_clusters_in_frame:
                    current_cluster_idx[0] = 0
                elif current_cluster_idx[0] < 0:
                    current_cluster_idx[0] = num_clusters_in_frame - 1

                # Get the specific cluster to display
                info = all_pcds_in_frame[0][current_cluster_idx[0]]
                vis.add_geometry(info['pcd']) # Add only the selected cluster's pcd
                rendered_pcds.append(info)

                print(f"\n--- Frame: {frame_num}, Cluster View [{current_cluster_idx[0]+1}/{num_clusters_in_frame}] ---")
                print(f"  Displaying Cluster ID: {info['id']}")
                print(f"  Points: {info['num_points']}")
                print(f"  True Label: {info['true_label']}")
                print(f"  Predicted Label: {info['predicted_label']}")
                print(f"  Status: {info['label_str']}")
                print(f"  File: {os.path.basename(info['path'])}")
            else:
                print(f"\n--- Frame: {frame_num}, Cluster View ---")
                print("  No clusters available in this frame.")

        print(f"Progress: Frame {frame_idx+1}/{len(frames)}")
        
        # Set initial view
        view_control = vis.get_view_control()
        front = [1, -1, 0] # Adjust as needed
        up = [0, 0, 1]     # Assuming Z is up
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(0.8) # Adjust zoom level

        return True

    # --- Key Callbacks ---
    def next_frame_callback(vis):
        current_frame_idx[0] = (current_frame_idx[0] + 1) % len(frames)
        current_cluster_idx[0] = 0 # Reset cluster index when changing frame
        update_visualization(current_frame_idx[0])
        return False

    def prev_frame_callback(vis):
        current_frame_idx[0] = (current_frame_idx[0] - 1 + len(frames)) % len(frames)
        current_cluster_idx[0] = 0 # Reset cluster index
        update_visualization(current_frame_idx[0])
        return False

    def next_cluster_callback(vis):
        if view_mode[0] == "cluster" and all_pcds_in_frame[0]:
            num_clusters = len(all_pcds_in_frame[0])
            current_cluster_idx[0] = (current_cluster_idx[0] + 1) % num_clusters
            update_visualization(current_frame_idx[0]) # Redraw with the new cluster
        return False

    def prev_cluster_callback(vis): # Added previous cluster functionality
        if view_mode[0] == "cluster" and all_pcds_in_frame[0]:
            num_clusters = len(all_pcds_in_frame[0])
            current_cluster_idx[0] = (current_cluster_idx[0] - 1 + num_clusters) % num_clusters
            update_visualization(current_frame_idx[0]) # Redraw with the new cluster
        return False

    def toggle_view_mode(vis):
        if view_mode[0] == "frame":
            view_mode[0] = "cluster"
            current_cluster_idx[0] = 0 # Start from first cluster
        else:
            view_mode[0] = "frame"
        update_visualization(current_frame_idx[0]) # Redraw with the new mode
        return False

    def quit_callback(vis):
        print("Exiting.")
        vis.destroy_window()
        return False

    # Register callbacks
    vis.register_key_callback(ord('N'), next_frame_callback)
    vis.register_key_callback(ord(' '), next_frame_callback) # Space also advances frame
    vis.register_key_callback(ord('B'), prev_frame_callback)
    vis.register_key_callback(ord('T'), toggle_view_mode)
    vis.register_key_callback(ord('C'), next_cluster_callback) # Next cluster
    vis.register_key_callback(ord('V'), prev_cluster_callback) # Previous cluster (using V)
    vis.register_key_callback(ord('Q'), quit_callback)

    # Initial visualization load
    update_visualization(current_frame_idx[0])

    # --- Print Controls ---
    print("\n--- Navigation Controls ---")
    print("  Space or N: Next frame")
    print("  B: Previous frame")
    print("  T: Toggle between Full Frame and Single Cluster view")
    print("  C: Next cluster (in cluster view)")
    print("  V: Previous cluster (in cluster view)")
    print("  Q: Quit")
    print("  Mouse: Rotate/Pan/Zoom")
    print("-------------------------\n")

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud classification predictions.')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the root directory containing the .bin point cloud cluster files.')
    parser.add_argument('--predictions_csv', type=str, required=True, 
                        help='Path to the predictions CSV file (e.g., predictions_val.csv).')
    parser.add_argument('--split', type=str, required=True, 
                        help='Path to the dataset split file (e.g., val_split.txt) corresponding to the predictions CSV.')
    
    args = parser.parse_args()

    # 1. Load predictions
    predictions_df = load_predictions(args.predictions_csv)
    if predictions_df is None:
        return

    # 2. Load split file
    split_name = os.path.basename(args.split).replace('.txt', '') # For window title
    split_filenames_full_path = load_split_file(args.split) # Get ordered list from split file
    if split_filenames_full_path is None:
        return
        
    # Extract basenames from the split file list for mapping
    split_basenames = [os.path.basename(f) for f in split_filenames_full_path]

    # 3. Map predictions to filenames
    prediction_mapping = map_predictions_to_files(split_basenames, predictions_df)
    if prediction_mapping is None:
        return

    # 4. Find all relevant .bin files in the data directory based on the split file
    # Construct full paths based on data_dir and the list from the split file
    # Assuming the split file contains relative paths or just basenames that need to be found
    print(f"Searching for .bin files listed in {args.split} within {args.data_dir}...")
    bin_files_to_visualize = []
    
    # Check if split file contains absolute paths or needs searching
    is_absolute = os.path.isabs(split_filenames_full_path[0])
    if is_absolute:
         bin_files_to_visualize = [f for f in split_filenames_full_path if os.path.exists(f) and os.path.basename(f) in prediction_mapping]
    else: # Assume relative paths or basenames, search within data_dir
        files_found_in_dir = {os.path.basename(f): os.path.join(root, f) 
                              for root, _, files in os.walk(args.data_dir) for f in files if f.endswith('.bin')}
        
        for fname in split_filenames_full_path: # Iterate through paths/names from split file
             basename = os.path.basename(fname)
             if basename in files_found_in_dir and basename in prediction_mapping:
                 bin_files_to_visualize.append(files_found_in_dir[basename]) # Use the full path found
             # elif basename in prediction_mapping: # Enable this if you want to warn about missing files
             #     print(f"Warning: File '{basename}' from split not found in data directory.")


    print(f"Found {len(bin_files_to_visualize)} .bin files corresponding to the split and predictions.")
    if not bin_files_to_visualize:
         print("Error: No matching .bin files found. Check data_dir and split file paths.")
         return

    # 5. Group files by frame
    frames = group_by_frame(bin_files_to_visualize)
    if not frames:
        print("Error: Could not group any files into frames. Check filename format.")
        return

    # 6. Start visualization
    visualize_predictions(frames, prediction_mapping, split_name)

if __name__ == "__main__":
    main() 