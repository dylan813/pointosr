cmake_minimum_required(VERSION 3.0.2)
project(pointosr_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  pointosr
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  PointCloud2Array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime
)

catkin_install_python(PROGRAMS
  scripts/multiplexer_node.py
  scripts/inference_node.py
  scripts/count_clusters.py
  scripts/update_val_batch_size.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/inference.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
