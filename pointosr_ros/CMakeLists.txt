cmake_minimum_required(VERSION 3.0.2)
project(pointosr_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  pointosr
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/test_classification_node.py
  scripts/count_clusters.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
