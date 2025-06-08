# An array of PointCloud2 messages
std_msgs/Header header
sensor_msgs/PointCloud2[] clouds
string[] topic_names # To keep track of the original source topic for each cloud 