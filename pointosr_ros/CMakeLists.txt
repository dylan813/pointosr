cmake_minimum_required(VERSION 3.0.2)
project(pointosr_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  pointosr
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/classification_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
