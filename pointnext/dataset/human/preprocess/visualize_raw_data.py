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

def visualize_frames(frames):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Frame Visualization", width=1024, height=768)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)
    
    current_idx = [0]
    
    def update_visualization(idx):
        if 0 <= idx < len(frames):
            vis.clear_geometries()
            vis.add_geometry(coordinate_frame)
            
            frame_num, clusters = frames[idx]
            
            total_points = 0
            cluster_info = []
            
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
                
                vis.add_geometry(pcd)
            
            print(f"\nFrame: {frame_num}")
            print(f"Clusters: {len(clusters)}")
            print(f"Total points: {total_points}")
            print("Cluster details:")
            for cluster_id, num_points in sorted(cluster_info):
                print(f"  Cluster {cluster_id}: {num_points} points")
            print(f"Progress: {idx+1}/{len(frames)}")
            
            view_control = vis.get_view_control()
            front = [1, -1, 0]
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
        update_visualization(current_idx[0])
        return False
    
    def prev_callback(vis):
        current_idx[0] -= 1
        if current_idx[0] < 0:
            current_idx[0] = len(frames) - 1
        update_visualization(current_idx[0])
        return False
    
    def quit_callback(vis):
        vis.destroy_window()
        return False
    
    vis.register_key_callback(ord('N'), next_callback)
    vis.register_key_callback(ord(' '), next_callback)
    vis.register_key_callback(ord('B'), prev_callback)
    vis.register_key_callback(ord('Q'), quit_callback)
    
    update_visualization(current_idx[0])
    
    print("\nNavigation Controls:")
    print("  Space or N: Next frame")
    print("  B: Previous frame")
    print("  Q: Quit")
    print("  Mouse: Rotate/Pan/Zoom")
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud frames')
    parser.add_argument('--data', type=str, required=True, help='Path to point cloud data directory')
    parser.add_argument('--frame', type=int, help='Specific frame to start with (optional)')
    args = parser.parse_args()
    
    bin_files = []
    for root, _, files in os.walk(args.data):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    
    frames = group_by_frame(bin_files)
    
    start_idx = 0
    if args.frame is not None:
        for i, (frame_num, _) in enumerate(frames):
            if frame_num == args.frame:
                start_idx = i
                break
    
    frames = frames[start_idx:] + frames[:start_idx]
    
    visualize_frames(frames)

if __name__ == "__main__":
    main()