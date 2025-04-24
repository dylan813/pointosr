import numpy as np
import open3d as o3d
import pandas as pd
import os
import argparse
import struct
import time
import re
from collections import defaultdict

def load_bin(file_path):
    """Load a binary point cloud file."""
    try:
        points = np.fromfile(file_path, dtype=np.float32)
        # Reshape assuming each point has x, y, z, i coordinates
        if len(points) % 4 == 0:
            points = points.reshape(-1, 4)
            return points
        # If not divisible by 4, try 3 (x, y, z)
        elif len(points) % 3 == 0:
            points = points.reshape(-1, 3)
            # Add a dummy intensity channel
            intensity = np.zeros((points.shape[0], 1))
            points = np.concatenate([points, intensity], axis=1)
            return points
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return np.zeros((1, 4))

def extract_frame_number(filename):
    """Extract frame number from filename pattern: cluster_frame_xxxxxx_cluster_x.bin"""
    match = re.search(r'cluster_frame_(\d+)_cluster_\d+\.bin', filename)
    if match:
        return int(match.group(1))
    return 0  # Default if pattern doesn't match

def extract_cluster_id(filename):
    """Extract cluster ID from filename pattern: cluster_frame_xxxxxx_cluster_x.bin"""
    match = re.search(r'cluster_frame_\d+_cluster_(\d+)\.bin', filename)
    if match:
        return int(match.group(1))
    return 0  # Default if pattern doesn't match

def group_by_frame(files_with_predictions):
    """Group files by frame number"""
    frames = defaultdict(list)
    for file_path, prediction in files_with_predictions:
        filename = os.path.basename(file_path)
        frame_num = extract_frame_number(filename)
        frames[frame_num].append((file_path, prediction))
    
    # Sort frames by frame number
    sorted_frames = sorted(frames.items())
    return sorted_frames

def visualize_frames(frames):
    """Visualize all clusters in each frame together"""
    # Create visualization window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Frame Visualization", width=1024, height=768)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)
    
    # Current frame index
    current_idx = [0]  # Use list for mutable reference
    
    def update_visualization(idx):
        if 0 <= idx < len(frames):
            # Clear previous point clouds
            vis.clear_geometries()
            vis.add_geometry(coordinate_frame)
            
            # Get current frame
            frame_num, clusters = frames[idx]
            
            # Load and display all clusters in this frame
            human_count = 0
            misc_count = 0
            
            for file_path, prediction in clusters:
                # Load point cloud
                points = load_bin(file_path)
                
                # Create point cloud object
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                
                # Color based on classification
                if prediction.lower() == 'human':
                    colors = np.zeros((len(points), 3))
                    colors[:, 0] = 1.0  # Red for humans
                    human_count += 1
                else:
                    colors = np.zeros((len(points), 3))
                    colors[:, 2] = 1.0  # Blue for misc
                    misc_count += 1
                
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Add to visualizer
                vis.add_geometry(pcd)
            
            # Print frame info
            print(f"\nFrame: {frame_num}")
            print(f"Clusters: {len(clusters)} (Human: {human_count}, Misc: {misc_count})")
            print(f"Progress: {idx+1}/{len(frames)}")
            
            # Set camera view
            view_control = vis.get_view_control()
            front = [1, -1, 0.5]
            up = [0, 0, 1]
            view_control.set_front(front)
            view_control.set_up(up)
            view_control.set_zoom(0.8)
            view_control.rotate(0, 30)  # Rotate 30 degrees around vertical axis
            
            return True
        return False
    
    # Key callbacks for navigation
    def next_callback(vis):
        current_idx[0] += 1
        if current_idx[0] >= len(frames):
            current_idx[0] = 0  # Loop back to start
        update_visualization(current_idx[0])
        return False
    
    def prev_callback(vis):
        current_idx[0] -= 1
        if current_idx[0] < 0:
            current_idx[0] = len(frames) - 1  # Loop to end
        update_visualization(current_idx[0])
        return False
    
    def quit_callback(vis):
        vis.destroy_window()
        return False
    
    # Register callbacks
    vis.register_key_callback(ord('N'), next_callback)  # Press 'N' for next
    vis.register_key_callback(ord(' '), next_callback)  # Press Space for next
    vis.register_key_callback(ord('P'), prev_callback)  # Press 'P' for previous
    vis.register_key_callback(ord('Q'), quit_callback)  # Press 'Q' to quit
    
    # Initial visualization
    update_visualization(current_idx[0])
    
    # Instructions
    print("\nNavigation Controls:")
    print("  Space or N: Next frame")
    print("  P: Previous frame")
    print("  Q: Quit")
    print("  Mouse: Rotate/Pan/Zoom")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud frames with classifications')
    parser.add_argument('--predictions', type=str, default='log/unlabeled_predictions/predictions.csv', help='Path to predictions CSV')
    parser.add_argument('--data_path', type=str, default='data/cluster_data', help='path to test data')
    parser.add_argument('--frame', type=int, help='Specific frame to start with (optional)')
    args = parser.parse_args()
    
    # Load predictions
    predictions_df = pd.read_csv(args.predictions)
    predictions_dict = dict(zip(predictions_df['Filename'], predictions_df['Predicted Class']))
    
    # Create list of files with predictions
    files_with_predictions = []
    for filename, prediction in predictions_dict.items():
        file_path = os.path.join(args.data_path, filename)
        if os.path.exists(file_path):
            files_with_predictions.append((file_path, prediction))
    
    if not files_with_predictions:
        print("No valid files found with predictions.")
        return
    
    # Group files by frame
    frames = group_by_frame(files_with_predictions)
    
    # If specific frame is provided, start with that frame
    start_idx = 0
    if args.frame is not None:
        for i, (frame_num, _) in enumerate(frames):
            if frame_num == args.frame:
                start_idx = i
                break
    
    # Reorder frames to start with the specified frame
    frames = frames[start_idx:] + frames[:start_idx]
    
    # Start visualization
    visualize_frames(frames)

if __name__ == "__main__":
    main()