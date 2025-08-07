#batch of classification results for a timestamp

#batch timing
std_msgs/Header header

#array of classified clusters
cluster_and_cls[] classified_clusters

#stats
int32 total_input_clusters
int32 total_processed_clusters
int32 human_count
int32 false_count
int32 processing_errors

#debugging
string[] processing_log
float64 processing_time_sec
string model_version
