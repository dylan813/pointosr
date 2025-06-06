cmake_minimum_required(VERSION 3.0.2)
project(pointosr)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

find_program(PYTHON_EXECUTABLE python)

add_custom_target(build_pointnet2_batch ALL
    COMMAND ${PYTHON_EXECUTABLE} setup.py build_ext --inplace
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pointnext/cpp/pointnet2_batch
)

install(FILES cmake/pythonpath.sh
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cmake)
