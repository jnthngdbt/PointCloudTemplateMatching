cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(PointCloudTemplateMatching)
find_package(PCL 1.9 REQUIRED COMPONENTS common features filters io search)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB SOURCES src/*.h src/*.cpp)
add_executable(PointCloudTemplateMatching ${SOURCES})

target_link_libraries(PointCloudTemplateMatching ${PCL_LIBRARIES})
