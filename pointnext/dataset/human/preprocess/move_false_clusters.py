import numpy as np
import open3d as o3d
import os
import argparse
import re
import shutil

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

def filter_small_clusters(bin_files, max_points=280):
    small_clusters = []
    
    for file_path in bin_files:
        points = load_bin(file_path)
        if len(points) <= max_points:
            small_clusters.append(file_path)
            filename = os.path.basename(file_path)
            frame_num = extract_frame_number(filename)
            cluster_id = extract_cluster_id(filename)
            print(f"Added: Frame {frame_num}, Cluster {cluster_id} - Points: {len(points)}")
    
    return small_clusters

def visualize_clusters(clusters, false_positive_dir, max_points):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Small Clusters Visualization (≤{max_points} points)", width=1024, height=768)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coordinate_frame)
    
    current_idx = [0]
    moved_files = []
    
    def update_visualization(idx):
        if 0 <= idx < len(clusters):
            vis.clear_geometries()
            vis.add_geometry(coordinate_frame)
            
            file_path = clusters[idx]
            filename = os.path.basename(file_path)
            frame_num = extract_frame_number(filename)
            cluster_id = extract_cluster_id(filename)
            
            points = load_bin(file_path)
            num_points = len(points)
            
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
            
            moved_status = "[MOVED]" if file_path in moved_files else ""
            print(f"\nCluster {idx+1}/{len(clusters)} {moved_status}")
            print(f"Frame: {frame_num}, Cluster ID: {cluster_id}")
            print(f"Points: {num_points}")
            print(f"File: {filename}")
            
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
        if current_idx[0] >= len(clusters):
            current_idx[0] = 0
        update_visualization(current_idx[0])
        return False
    
    def prev_callback(vis):
        current_idx[0] -= 1
        if current_idx[0] < 0:
            current_idx[0] = len(clusters) - 1
        update_visualization(current_idx[0])
        return False
    
    def mark_false_positive(vis):
        if 0 <= current_idx[0] < len(clusters):
            file_path = clusters[current_idx[0]]
            if file_path in moved_files:
                print("This file has already been marked as a false positive.")
                return False
                
            filename = os.path.basename(file_path)
            target_path = os.path.join(false_positive_dir, filename)
            
            os.makedirs(false_positive_dir, exist_ok=True)
            
            try:
                shutil.move(file_path, target_path)
                print(f"Marked as false positive: {filename}")
                print(f"Moved to: {target_path} (removed from original location)")
                moved_files.append(file_path)
            except Exception as e:
                print(f"Error moving file: {e}")
                
        return False
    
    def quit_callback(vis):
        print("\nSummary:")
        print(f"Total clusters viewed: {len(clusters)}")
        print(f"Clusters marked as false positives: {len(moved_files)}")
        
        if moved_files:
            print("\nFiles marked as false positives:")
            for file_path in moved_files:
                print(f"- {os.path.basename(file_path)}")
                
        vis.destroy_window()
        return False
    
    vis.register_key_callback(ord('N'), next_callback)
    vis.register_key_callback(ord(' '), next_callback)
    vis.register_key_callback(ord('B'), prev_callback)
    vis.register_key_callback(ord('F'), mark_false_positive)
    vis.register_key_callback(ord('Q'), quit_callback)
    
    update_visualization(current_idx[0])
    
    print(f"\nIndividual Cluster Visualization (≤{max_points} points)")
    print("\nNavigation Controls:")
    print("  Space or N: Next cluster")
    print("  B: Previous cluster")
    print("  F: Mark current cluster as false positive")
    print("  Q: Quit and show summary")
    print("  Mouse: Rotate/Pan/Zoom")
    
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize small point cloud clusters')
    parser.add_argument('--data', type=str, required=True, help='Path to point cloud data directory')
    parser.add_argument('--max_points', type=int, default=280, help='Maximum number of points for a cluster to be included (default: 280)')
    parser.add_argument('--start_cluster', type=int, default=0, help='Index of cluster to start with (default: 0)')
    parser.add_argument('--false_positive_dir', type=str, default='', help='Directory to move false positive clusters (default: data_dir/../false_clusters)')
    args = parser.parse_args()
    
    bin_files = []
    for root, _, files in os.walk(args.data):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    
    print(f"Found {len(bin_files)} total cluster files")
    
    small_clusters = filter_small_clusters(bin_files, args.max_points)
    print(f"Filtered down to {len(small_clusters)} clusters with ≤{args.max_points} points")
    
    if not small_clusters:
        print("No clusters found with the specified criteria.")
        return
    
    small_clusters.sort(key=lambda f: (extract_frame_number(os.path.basename(f)), 
                                      extract_cluster_id(os.path.basename(f))))
    
    start_idx = args.start_cluster
    if start_idx >= len(small_clusters):
        start_idx = 0
    elif start_idx < 0:
        start_idx = 0
    
    small_clusters = small_clusters[start_idx:] + small_clusters[:start_idx]
    
    false_positive_dir = args.false_positive_dir
    if not false_positive_dir:
        data_parent_dir = os.path.dirname(os.path.abspath(args.data))
        false_positive_dir = os.path.join(data_parent_dir, 'false_clusters')
    
    visualize_clusters(small_clusters, false_positive_dir, args.max_points)

if __name__ == "__main__":
    main() 