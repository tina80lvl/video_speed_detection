cmake_minimum_required(VERSION 3.21)
project(vspeed)

set(CMAKE_CXX_STANDARD 20)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/Cellar/opencv@2/2.4.13.7_12.reinstall/share/OpenCV/OpenCVConfig.cmake)
find_package(OpenCV REQUIRED)

add_executable(vspeed main.cpp)

set(MAIN_PROJECT_SRC_LIST main)
add_executable(main ${MAIN_PROJECT_SRC_LIST})

add_subdirectory(libs)
target_link_libraries(vspeed lib)
add_subdirectory(utils)
target_link_libraries(vspeed utils)
add_subdirectory(algo)
target_link_libraries(vspeed algo)
