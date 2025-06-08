cmake_minimum_required(VERSION 3.0.2)
project(pointosr)

## Find required packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

## Generate messages
add_message_files(
  FILES
  PointCloud2Array.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

# Declare Python package setup
catkin_python_setup()

## Specify catkin project info
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pointosr
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs message_runtime
#  DEPENDS system_lib
) 