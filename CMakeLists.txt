cmake_minimum_required(VERSION 3.0.2)
project(mulls_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
# set(CMAKE_BUILD_TYPE "Debug")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -g  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -g -O3")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(EIGEN3 3 REQUIRED)

#TEASER++
set(teaserpp_DIR /usr/local/teaserpp/lib/cmake/teaserpp)
find_package(teaserpp REQUIRED)
message(STATUS "TEASER include dir: ${teaserpp_INCLUDE_DIRS}")
add_definitions(-DTEASER_ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  netvlad_tf_mulls
  visualization_msgs
  jsk_rviz_plugins
  jsk_recognition_msgs
)


# OpenCV
#set(OpenCV_DIR /usr/local/opencv-3.3.1/share/OpenCV)
find_package(OpenCV 3.2 REQUIRED)
add_definitions(-DOPENCV_ON)


find_package(Pangolin REQUIRED)

find_package(PCL 1.7 QUIET)
message(STATUS "pcl include dir: ${PCL_INCLUDE_DIRS}")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# GFLAG (neccessary)
find_package(GFLAGS REQUIRED QUIET)
if(GFLAGS_FOUND)
message("GFLAGS [OK]:" ${GFLAGS_INCLUDE_DIRS}) 
endif(GFLAGS_FOUND)

#GLOG (neccessary)
find_package(GLOG REQUIRED QUIET)
if(GLOG_FOUND)
message("GLOG [OK]:" ${GLOG_INCLUDE_DIRS}) 
endif(GLOG_FOUND)

## CERES
set(Ceres_DIR /usr/local/ceres_1.14.0/lib/cmake/Ceres)
find_package(Ceres REQUIRED)
message(STATUS "Ceres include : ${CERES_INCLUDE_DIRS}")
if(CERES_FOUND)
  add_definitions(-DCERES_ON)
endif()

# FIND_PACKAGE(Sophus REQUIRED)
set(Sophus_INCLUDE_DIRS /usr/local/sophus/include)
set(Sophus_LIBRARIES /usr/local/sophus/lib/libSophus.so)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mulls_ros
#  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_conversion pcl_ros roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  include/mulls_ros
  include/mulls_ros/baseline_reg
  include/mulls_ros/mulls_feature_extraction
  include/mulls_ros/lidar_odometry
  include/mulls_ros/pgo
  include/mulls_ros/nav
  ${PROJECT_SOURCE_DIR}/ThirdParty/ELSED/src
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/ThirdParty/vikit_common/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${teaserpp_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/mulls_ros.cpp
# )
## Specify libraries to link a library or executable target against

add_executable(${PROJECT_NAME}_raw_node 
  src/mulls_ros_raw_node.cpp
  src/map_manager.cpp
  src/common_nav.cpp
  src/build_pose_graph.cpp
  src/graph_optimizer.cpp

  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/VisualFeatureExtractor.cc
  
  src/lidar_sparse_align/SparseLidarAlign.cc
  src/lidar_sparse_align/WeightFunction.cpp

  src/LidarFeatureExtractor.cpp
  src/MultiScanRegistration.cc
  src/BasicScanRegistration.cc
  src/CeresOptimizer.cc


  # src/lidar_odometry/odomEstimationClass.cpp
)

target_link_libraries(${PROJECT_NAME}_raw_node
  ${catkin_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  teaserpp::teaser_registration 
  teaserpp::teaser_io
  ${PROJECT_SOURCE_DIR}/ThirdParty/ELSED/lib/libelsed.so
  ${PROJECT_SOURCE_DIR}/ThirdParty/DBoW3/lib/libDBoW3.so
  ${PROJECT_SOURCE_DIR}/ThirdParty/vikit_common/lib/libvikit_common.so
  ${Sophus_LIBRARIES}
  -lpthread
)



add_executable(bev_dbow_loop_node
  src/bev_dbow_loop_node.cpp
  src/ORBextractor.cc
  src/ORBmatcher.cc
)
target_link_libraries(bev_dbow_loop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${PCL_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/ThirdParty/DBoW3/lib/libDBoW3.so
)
