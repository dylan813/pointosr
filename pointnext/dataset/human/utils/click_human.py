import numpy as np
import open3d as o3d
import os
import argparse
import re
from collections import defaultdict
from typing import List, Tuple

def load_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape(-1, 4)
    return points

def extract_frame_number(filename):
    match = re.search(r'cluster_frame_(\d+)_cluster_\d+\.bin', filename)
    if match:
        return int(match.group(1))
    return 0

def extract_cluster_id(filename):
    match = re.search(r'cluster_frame_\d+_cluster_(\d+)\.bin', filename)
    if match:
        return int(match.group(1))
    return 0

def group_by_frame(files):
    frames = defaultdict(list)
    for file_path in files:
        filename = os.path.basename(file_path)
        frame_num = extract_frame_number(filename)
        frames[frame_num].append(file_path)
    
    sorted_frames = sorted(frames.items())
    return sorted_frames

def load_split_file(split_file_path):
    with open(split_file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def visualize_frames(frames, output_file_path, split_name=None):
    # Use VisualizerWithKeyCallback for key bindings
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_title = "Frame Visualization"
    if split_name:
        window_title += f" - {split_name} split"
    vis.create_window(window_name=window_title, width=1024, height=768)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame) 
    
    current_idx = [0]
    current_cluster_idx = [0]
    view_mode = ["frame"]
    # Store all point clouds for the current frame with file paths
    # Needed to map current_cluster_idx back to file path for selection
    all_pcds_data: List[Tuple[int, o3d.geometry.PointCloud, str]] = [] 

    # Define a list of distinct colors for clusters (toned down)
    DISTINCT_COLORS = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
        [1.0, 0.5, 0.5],  # Pink
        [0.5, 0.5, 1.0],  # Lavender
        [0.8, 0.8, 0.2],  # Olive
        [0.8, 0.2, 0.8],  # Violet
        [0.7, 0.7, 0.7],  # Gray
        [0.5, 0.2, 0.2],  # Dark Red
        [0.2, 0.5, 0.2],  # Dark Green
        [0.2, 0.2, 0.5],  # Dark Blue
    ]

    def update_visualization(idx):
        # Removed nonlocal current_geometries_filepaths
        nonlocal all_pcds_data 
        if 0 <= idx < len(frames):
            vis.clear_geometries()
            vis.add_geometry(coordinate_frame) # Re-add coordinate frame
            # Removed current_geometries_filepaths.clear()
            all_pcds_data.clear()
            
            frame_num, clusters = frames[idx]
            
            total_points = 0
            cluster_info = []
            temp_pcds_data = [] 
            
            sorted_clusters = sorted(clusters, key=lambda fp: extract_cluster_id(os.path.basename(fp)))

            for cluster_index, file_path in enumerate(sorted_clusters): # Use enumerate for index
                points = load_bin(file_path)
                cluster_id = extract_cluster_id(os.path.basename(file_path))
                num_points = len(points)
                total_points += num_points
                cluster_info.append((cluster_id, num_points))
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                
                color_idx = cluster_index % len(DISTINCT_COLORS)
                color = DISTINCT_COLORS[color_idx]
                pcd.paint_uniform_color(color)
                
                temp_pcds_data.append((cluster_id, pcd, file_path)) 
            
            all_pcds_data = sorted(temp_pcds_data, key=lambda x: x[0]) 
            
            if view_mode[0] == "cluster":
                if all_pcds_data:
                    if current_cluster_idx[0] >= len(all_pcds_data):
                        current_cluster_idx[0] = 0
                    elif current_cluster_idx[0] < 0:
                        current_cluster_idx[0] = len(all_pcds_data) - 1
                    
                    cluster_id, pcd, file_path = all_pcds_data[current_cluster_idx[0]]
                    vis.add_geometry(pcd)
                    # Removed current_geometries_filepaths.append(file_path)
                    print(f"\nFrame: {frame_num}, Showing Cluster {cluster_id} ({current_cluster_idx[0]+1}/{len(all_pcds_data)})")
                    print(f"Cluster {cluster_id}: {len(pcd.points)} points")
                    print(f"File: {file_path}")
                else:
                    print(f"\nFrame: {frame_num}, No clusters available")
            else: # Frame view mode
                print(f"\nFrame: {frame_num} (Full View)")
                print(f"Clusters: {len(clusters)}")
                print(f"Total points: {total_points}")
                print("Cluster details:")
                for cluster_id, pcd, file_path in all_pcds_data:
                    vis.add_geometry(pcd)
                    # Removed current_geometries_filepaths.append(file_path)
                    print(f"  Cluster {cluster_id}: {len(pcd.points)} points ({os.path.basename(file_path)})")
            
            print(f"Progress: {frame_num}/{len(frames)}")
            
            view_control = vis.get_view_control()
            front = [1, -1, 0]
            up = [0, 0, 1]
            view_control.set_front(front)
            view_control.set_up(up)
            view_control.set_zoom(0.8)
            view_control.rotate(0, 30)
            
            return True
        return False
    
    # --- Callbacks ---
    
    def next_callback(vis):
        current_idx[0] += 1
        if current_idx[0] >= len(frames):
            current_idx[0] = 0
        current_cluster_idx[0] = 0
        update_visualization(current_idx[0])
        return False
    
    def prev_callback(vis):
        current_idx[0] -= 1
        if current_idx[0] < 0:
            current_idx[0] = len(frames) - 1
        current_cluster_idx[0] = 0
        update_visualization(current_idx[0])
        return False
    
    def next_cluster_callback(vis):
        if view_mode[0] == "cluster":
            current_cluster_idx[0] += 1
            update_visualization(current_idx[0])
        return False
    
    def toggle_view_mode(vis):
        if view_mode[0] == "frame":
            view_mode[0] = "cluster"
            current_cluster_idx[0] = 0
        else:
            view_mode[0] = "frame"
        update_visualization(current_idx[0])
        return False
    
    # Renamed picking_callback to select_callback, triggered by key 'S'
    def select_callback(vis):
        nonlocal all_pcds_data, current_cluster_idx, view_mode, output_file_path
        
        if view_mode[0] == "cluster":
            if all_pcds_data and 0 <= current_cluster_idx[0] < len(all_pcds_data):
                cluster_id, _, selected_file_path = all_pcds_data[current_cluster_idx[0]]
                base_name = os.path.basename(selected_file_path)
                print(f"\n--- Key 'S' Pressed ---")
                print(f"  Cluster ID: {cluster_id}")
                print(f"  Cluster File: {base_name}")

                # Check if the file path is already recorded
                already_recorded = False
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, 'r') as f:
                            existing_paths = {line.strip() for line in f}
                            if selected_file_path in existing_paths:
                                already_recorded = True
                except Exception as e:
                    print(f"  Warning: Could not read output file {output_file_path} to check for duplicates: {e}")

                if not already_recorded:
                    try:
                        with open(output_file_path, 'a') as f:
                            f.write(selected_file_path + '\n')
                        print(f"  Recorded to: {output_file_path}")
                    except Exception as e:
                        print(f"  Error writing to file {output_file_path}: {e}")
                else:
                    print(f"  Already recorded in: {output_file_path}")
                    
                print(f"-------------------------")
            else:
                print("\nCannot select: No cluster currently displayed or index out of bounds.")
        else:
            print("\nPlease switch to cluster view mode ('T') and cycle ('C') to the desired cluster before selecting ('S').")
            
        return False # Keep callback registered

    def quit_callback(vis):
        vis.destroy_window()
        return False
    
    # Register key callbacks
    vis.register_key_callback(ord('N'), next_callback)
    vis.register_key_callback(ord(' '), next_callback)
    vis.register_key_callback(ord('B'), prev_callback)
    vis.register_key_callback(ord('Q'), quit_callback)
    vis.register_key_callback(ord('T'), toggle_view_mode)
    vis.register_key_callback(ord('C'), next_cluster_callback)
    vis.register_key_callback(ord('S'), select_callback) # Register 'S' for selection

    # Removed register_selection_changed_callback
    
    update_visualization(current_idx[0])
    
    print("\nNavigation Controls:")
    print("  Space or N: Next frame")
    print("  B: Previous frame")
    print("  T: Toggle between full frame and individual cluster view")
    print("  C: Next cluster (in cluster view mode)")
    print("  S: Select current cluster (in cluster view mode) to record its file path")
    print("  Q: Quit")
    print("  Mouse: Rotate/Pan/Zoom")
    
    vis.run()
    # Destroy window is handled by quit_callback

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud frames with cluster clicking')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to point cloud data directory')
    parser.add_argument('--frame', type=int, help='Specific frame to start with (optional)')
    parser.add_argument('--split', type=str, help='Path to dataset split')
    parser.add_argument('--output_file', type=str, default='clicked_clusters.txt', help='File to record clicked cluster paths (default: clicked_clusters.txt)')
    args = parser.parse_args()
    
    split_filenames = None
    split_name = None
    
    if args.split:
        split_name = os.path.basename(args.split).replace('_split.txt', '')
        split_filenames = load_split_file(args.split)
        print(f"Loaded {len(split_filenames)} files from {args.split}")
    
    bin_files = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                filename_only = os.path.basename(file_path)
                
                if split_filenames is None or filename_only in split_filenames:
                    bin_files.append(file_path)
    
    if split_filenames is not None:
        print(f"Found {len(bin_files)} matching files from the {split_name} split using filenames: {', '.join(list(split_filenames)[:5])}...")
    else:
        print(f"Found {len(bin_files)} .bin files in {args.data_dir}")
    
    frames = group_by_frame(bin_files)
    
    if not frames:
        print("No valid frames found. Please check your data path and dataset split.")
        return
    
    start_idx = 0
    if args.frame is not None:
        for i, (frame_num, _) in enumerate(frames):
            if frame_num == args.frame:
                start_idx = i
                break
        print(f"Starting at frame {args.frame} (index {start_idx})")
    else:
        print("Starting at the first frame (index 0)")

    if start_idx > 0:
        frames = frames[start_idx:] + frames[:start_idx]
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Clicked cluster file paths will be saved to: {args.output_file}")
    
    visualize_frames(frames, args.output_file, split_name)

if __name__ == "__main__":
    main()