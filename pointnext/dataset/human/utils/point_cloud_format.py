import numpy as np

dt = np.float32     #float32 or float64
filepath = '/home/dylanleo/Documents/point_osr/data/human/human_clusters/cluster_frame_000001_cluster_0.bin' 

try:
    raw_data = np.fromfile(filepath, dtype=dt)
    
    print(f"Read {raw_data.size} values.")
    
    if raw_data.size % 4 == 0:
        num_points = raw_data.size // 4
        print(f"{num_points} points with 4 values each.")
        
        points = raw_data.reshape(-1, 4)
        print("First 5 points:\n", points[:5, :])
        
        print("X range:", points[:, 0].min(), points[:, 0].max())
        print("Y range:", points[:, 1].min(), points[:, 1].max())
        print("Z range:", points[:, 2].min(), points[:, 2].max())
        print("Intensity range:", points[:, 3].min(), points[:, 3].max())
        
    else:
        print(f"Total number of values ({raw_data.size}) is not divisible by 4. Check format/dtype.")
        print("First 20 values:", raw_data[:20])

except Exception as e:
    print(f"Error reading file: {e}")
