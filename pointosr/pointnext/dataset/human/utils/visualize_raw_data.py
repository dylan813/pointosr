import numpy as np
import open3d as o3d
import os
import argparse
import re
from collections import defaultdict

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

def visualize_frames(frames, split_name=None):
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
    
    def update_visualization(idx):
        if 0 <= idx < len(frames):
            vis.clear_geometries()
            vis.add_geometry(coordinate_frame)
            
            frame_num, clusters = frames[idx]
            
            total_points = 0
            cluster_info = []
            all_pcds = []
            
            for file_path in clusters:
                points = load_bin(file_path)
                cluster_id = extract_cluster_id(os.path.basename(file_path))
                num_points = len(points)
                total_points += num_points
                cluster_info.append((cluster_id, num_points))
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                
                intensities = points[:, 3]
                
                if len(intensities) > 0:
                    min_intensity = np.min(intensities)
                    max_intensity = np.max(intensities)
                    intensity_range = max_intensity - min_intensity
                    
                    if intensity_range > 0:
                        normalized_intensities = (intensities - min_intensity) / intensity_range
                    else:
                        normalized_intensities = np.zeros_like(intensities)
                    
                    colors = np.zeros((len(points), 3))
                    colors[:, 0] = normalized_intensities
                    colors[:, 1] = normalized_intensities
                    colors[:, 2] = normalized_intensities
                else:
                    colors = np.zeros((len(points), 3))
                
                pcd.colors = o3d.utility.Vector3dVector(colors)
                all_pcds.append((cluster_id, pcd))
                
                if view_mode[0] == "frame":
                    vis.add_geometry(pcd)
            
            all_pcds.sort(key=lambda x: x[0])
            
            if view_mode[0] == "cluster":
                if all_pcds:
                    if current_cluster_idx[0] >= len(all_pcds):
                        current_cluster_idx[0] = 0
                    elif current_cluster_idx[0] < 0:
                        current_cluster_idx[0] = len(all_pcds) - 1
                    
                    cluster_id, pcd = all_pcds[current_cluster_idx[0]]
                    vis.add_geometry(pcd)
                    print(f"\nFrame: {frame_num}, Showing Cluster {cluster_id} ({current_cluster_idx[0]+1}/{len(all_pcds)})")
                    print(f"Cluster {cluster_id}: {len(pcd.points)} points")
                else:
                    print(f"\nFrame: {frame_num}, No clusters available")
            else:
                print(f"\nFrame: {frame_num} (Full View)")
                print(f"Clusters: {len(clusters)}")
                print(f"Total points: {total_points}")
                print("Cluster details:")
                for cluster_id, num_points in sorted(cluster_info):
                    print(f"  Cluster {cluster_id}: {num_points} points")
            
            print(f"Progress: {frame_num}/{len(frames)-1}")
            
            view_control = vis.get_view_control()
            front = [1, -1, 0.5]
            up = [0, 0, 1]
            view_control.set_front(front)
            view_control.set_up(up)
            view_control.set_zoom(0.8)
            view_control.rotate(0, 30)
            
            return True
        return False
    
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
    
    def quit_callback(vis):
        vis.destroy_window()
        return False
    
    vis.register_key_callback(ord('N'), next_callback)
    vis.register_key_callback(ord(' '), next_callback)
    vis.register_key_callback(ord('B'), prev_callback)
    vis.register_key_callback(ord('Q'), quit_callback)
    vis.register_key_callback(ord('T'), toggle_view_mode)
    vis.register_key_callback(ord('C'), next_cluster_callback)
    
    update_visualization(current_idx[0])
    
    print("\nNavigation Controls:")
    print("  Space or N: Next frame")
    print("  B: Previous frame")
    print("  T: Toggle between full frame and individual cluster view")
    print("  C: Next cluster (in cluster view mode)")
    print("  Q: Quit")
    print("  Mouse: Rotate/Pan/Zoom")
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud frames')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to point cloud data directory')
    parser.add_argument('--frame', type=int, help='Specific frame to start with (optional)')
    parser.add_argument('--split', type=str, help='Path to dataset split')
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
                
                if split_filenames is None or file in split_filenames:
                    bin_files.append(file_path)
    
    if split_filenames is not None:
        print(f"Found {len(bin_files)} matching files from the {split_name} split")
    
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
    
    frames = frames[start_idx:] + frames[:start_idx]
    
    visualize_frames(frames, split_name)

if __name__ == "__main__":
    main()