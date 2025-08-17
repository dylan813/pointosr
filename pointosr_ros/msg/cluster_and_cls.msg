#point cloud data with classification results

#timing and frame information
std_msgs/Header header

sensor_msgs/PointCloud2 pointcloud

#classification results
string class_name
float32 confidence
bool is_human

#OSR (Open Set Recognition) results
bool is_ood
float32 energy_score
float32 cosine_score
float32 fused_score
float32 ood_confidence

#metadata
int32 original_cluster_index
bool processing_success
string error_message

#geometric info
geometry_msgs/Point centroid
float32 cluster_size
