#point cloud data with classification results

#timing and frame information
std_msgs/Header header

sensor_msgs/PointCloud2 pointcloud

#classification results
string class_name
float32 confidence
bool is_human

#metadata
int32 original_cluster_index
bool processing_success
string error_message

#geometric info
geometry_msgs/Point centroid
float32 cluster_size
