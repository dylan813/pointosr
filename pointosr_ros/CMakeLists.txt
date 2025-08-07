cmake_minimum_required(VERSION 3.0.2)
project(pointosr_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  pointosr
)

add_message_files(
  FILES
  cluster_and_cls.msg
  classification_batch.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_runtime
  pointosr
)

catkin_install_python(PROGRAMS
  scripts/classification_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
